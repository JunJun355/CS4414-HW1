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

// 4, 5, 12920, 26743, 6, 0, 3, 68740, 40982, 63176,

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
    // Should be precisely 32 bytes, since int+short+short+T is 4+2+2+24
    int idx = 0;
    unsigned short left_idx = -1;
    unsigned short right_idx = -1;
    T* embedding;
};

template <typename T>
struct Node
{
    //janky way to return a tree instead of a node
    std::array<SubNode<T>, 40000> nodes;
    int mid = 0;
    // static query for comparisons
    static T queryEmbedding;
};

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;

template <typename T>
unsigned short buildKD_aux(
    std::vector<std::pair<T,int>>& items,
    Node<T>* root,
    int start,
    int end,
    int depth
) {
    if (start == end) return -1;
    int d = Embedding_T<T>::Dim();
    int split_dim = (depth) % d;
    
    int mid = (start + end) / 2;

    std::nth_element(items.begin() + start, items.begin() + mid, items.begin() + end, [split_dim, d](const std::pair<T,int>& a, const std::pair<T,int>& b) {
        float diff = getDiffAtCoor(a.first, b.first, split_dim);
        if (diff != 0) return diff < 0;

        for (int i=split_dim + 1; i != split_dim; i = (i + 1) % d) {
            diff = getDiffAtCoor(a.first, b.first, i);
            if (diff != 0) return diff < 0;
        }
        return false;
    });

    SubNode<T>* curr = &(root->nodes[mid]);

    curr->embedding = &(items[mid].first);
    curr->idx = items[mid].second;

    if (depth < 1) {
        auto fleft = std::async(std::launch::async, [&items, root, start, mid, depth]() {
            return buildKD_aux(items, root, start, mid, depth + 1);
        });

        curr->right_idx = buildKD_aux(items, root, mid + 1, end, depth + 1);
        curr->left_idx= fleft.get();
    }
    else {
        curr->left_idx = buildKD_aux(items, root, start, mid, depth + 1);
        curr->right_idx = buildKD_aux(items, root, mid + 1, end, depth + 1);
    }

    return mid;
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
    Node<T>* root = new Node<T>();
    root->mid = items.size() / 2;

    buildKD_aux(items, root, 0, (int) items.size(), depth);
    return root;
}


template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
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

// int asdf=0;
template <typename T>
void knnSearch_aux(Node<T> *root, int idx, int depth, int K, MaxHeap &heap) {
    // asdf++;
    int split_dim = depth % Embedding_T<T>::Dim();

    SubNode<T>* node = &(root->nodes[idx]);
    
    float qdist = Embedding_T<T>::distance(*(node->embedding), Node<T>::queryEmbedding);
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
    unsigned short close, far;
    if (getDiffAtCoor<T>(*(node->embedding), Node<T>::queryEmbedding, split_dim) > 0) {
        close = node->left_idx;
        far = node->right_idx;
    }
    else {
        close = node->right_idx;
        far = node->left_idx;
    }
    if (close != 65535) knnSearch_aux(root, close, depth + 1, K, heap);
    if ((int) heap.size() < K || std::abs(getDiffAtCoor<T>(*(node->embedding), Node<T>::queryEmbedding, split_dim)) < heap.top().first) {
        if (far != 65535) knnSearch_aux(root, far, depth + 1, K, heap);
    }
}

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
void knnSearch(Node<T> *root,
               int depth,
               int K,
               MaxHeap &heap)
{
    knnSearch_aux(root, root->mid, depth, K, heap);
}