#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>
#include <thread>
#include <future>
#include <memory>

template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b) { return std::abs(a - b); }
};


// dynamic vector: runtime-D (global, set once at startup)
inline size_t& runtime_dim() {
    static size_t d = 0;
    return d;
}

// variable-size vector: N-D
template <>
struct Embedding_T<std::vector<float>>
{
    static size_t Dim() { return runtime_dim(); }
    
    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        float s = 0;
        for (size_t i = 0; i < Dim(); ++i)
        {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);
    }
};


// extract the “axis”-th coordinate or the scalar itself
template<typename T>
constexpr float getCoordinate(T const &e, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return e;          // scalar case
    } else {
        return e[axis];    // vector case
    }
}

template<typename T>
constexpr float getDiffAtCoor(T const &a, T const &b, size_t axis) {
    if constexpr (std::is_same_v<T, float>) {
        return a - b;          // scalar case
    } else {
        return a[axis] - b[axis];    // vector case
    }
}

// KD-tree node
template <typename T>
struct SubNode {
    int idx = 0;
    int left_idx = -1;
    int right_idx = -1;
    T embedding;
};

template <typename T>
struct Node
{
    //janky way to return a tree instead of a node
    // std::array<SubNode<T>, 40000> nodes;

    T embedding;
    int idx;
    Node *left = nullptr;
    Node *right = nullptr;

    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;

template <typename T>
Node<T>* buildKD_aux(
    std::vector<std::pair<T,int>>& items,
    std::vector<Node<T>*>& node_pool,
    int start,
    int end,
    int depth
) {
    // if (depth == 2) return nullptr;
    if (start == end) return nullptr;
    int d = Embedding_T<T>::Dim();
    int split_dim = (depth) % d;
    
    int mid = (start + end - 1) / 2;

    std::nth_element(items.begin() + start, items.begin() + mid, items.begin() + end, [split_dim, d](const std::pair<T,int>& a, const std::pair<T,int>& b) {
        // float diff = getCoordinate(a.first, split_dim) - getCoordinate(b.first, split_dim);
        float diff = getDiffAtCoor(a.first, b.first, split_dim);
        if (diff != 0) return diff < 0;
        for (int i=split_dim + 1; i < d; i++) {
            diff = getDiffAtCoor(a.first, b.first, i);
            // diff = getCoordinate(a.first, i) - getCoordinate(b.first, i);
            if (diff == 0) continue;
            return diff < 0;
        }
        for (int i=0; i < split_dim; i++) {
            diff = getDiffAtCoor(a.first, b.first, i);
            // diff = getCoordinate(a.first, i) - getCoordinate(b.first, i);
            if (diff == 0) continue;
            return diff < 0;
        }
        return false;
    });

    Node<T>* root = node_pool[mid];

    root->embedding = items[mid].first;
    root->idx = items[mid].second;

    if (depth < 1) {
        auto fleft = std::async(std::launch::async, [&items, &node_pool, start, mid, depth]() {
            return buildKD_aux(items, node_pool, start, mid, depth + 1);
        });

        root->right = buildKD_aux(items, node_pool, mid + 1, end, depth + 1);
        root->left = fleft.get();
    }
    else {
        root->left = buildKD_aux(items, node_pool, start, mid, depth + 1);
        root->right = buildKD_aux(items, node_pool, mid + 1, end, depth + 1);
    }

    return root;
}


/**
 * Builds a KD-tree from a vector of items,
 * where each item consists of an embedding and its associated index.
 * The splitting dimension is chosen based on the current depth.
 *
 * @param items A reference to a vector of pairs, each containing an embedding (Embedding_T)
 *              and an integer index.
 * @param depth The current depth in the tree, used to determine the splitting dimension (default is 0).
 * @return A pointer to the root node of the constructed KD-tree.
 */
// Build a balanced KD‐tree by splitting on median at each level.
template <typename T>
Node<T>* buildKD(std::vector<std::pair<T,int>>& items, int depth = 0)
{
    std::vector<Node<T>*> node_pool(items.size());
    for (int i=0; i<(int)items.size(); i++) node_pool[i] = new Node<T>();

    return buildKD_aux(items, node_pool, 0, (int) items.size(), depth);
}

template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


/**
 * @brief Alias for a pair consisting of a float and an int.
 *
 * Typically used to represent a priority queue item where the float
 * denotes the priority (the distance of an embedding to the query embedding) and the int
 * represents an associated index of the embedding.
 */
using PQItem = std::pair<float, int>;


/**
 * @brief Alias for a max-heap priority queue of PQItem elements.
 *
 * This type uses std::priority_queue with PQItem as the value type,
 * std::vector<PQItem> as the underlying container, and std::less<PQItem>
 * as the comparison function, resulting in a max-heap behavior.
 */
using MaxHeap = std::priority_queue<
    PQItem,
    std::vector<PQItem>,
    std::less<PQItem>>;

/**
 * @brief Performs a k-nearest neighbors (k-NN) search on a KD-tree.
 *
 * This function recursively traverses the KD-tree starting from the given node,
 * searching for the K nearest neighbors to a target point. The results are maintained
 * in a max-heap, and an optional epsilon parameter can be used to allow for approximate
 * nearest neighbor search.
 *
 * @param node Pointer to the current node in the KD-tree.
 * @param depth Current depth in the KD-tree (used to determine splitting axis).
 * @param K Number of nearest neighbors to search for.
 * @param epsilon Approximation factor for the search (0 for exact search).
 * @param heap Reference to a max-heap that stores the current K nearest neighbors found.
 */
template <typename T>
void knnSearch(Node<T> *node,
               int depth,
               int K,
               MaxHeap &heap)
{
    if (node == nullptr) return;

    int split_dim = depth % Embedding_T<T>::Dim();
    
    float qdist = Embedding_T<T>::distance(node->embedding, Node<T>::queryEmbedding);
    if ((int) heap.size() < K) {
        heap.push(PQItem(qdist, node->idx));
    }
    else {
        float hdist = heap.top().first;
        if (hdist > qdist) {
            heap.pop();
            heap.push(PQItem(qdist, node->idx));
        }
    }
    Node<T> *close, *far;
    if (getCoordinate<T>(node->embedding, split_dim) > getCoordinate<T>(Node<T>::queryEmbedding, split_dim)) {
        close = node->left;
        far = node->right;
    }
    else {
        close = node->right;
        far = node->left;
    }
    knnSearch(close, depth + 1, K, heap);
    if ((int) heap.size() < K || std::abs(getCoordinate<T>(Node<T>::queryEmbedding, split_dim) - getCoordinate<T>(node->embedding, split_dim)) < heap.top().first) {
        knnSearch(far, depth + 1, K, heap);
    }
    return;
}