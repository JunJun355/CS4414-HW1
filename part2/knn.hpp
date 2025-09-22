#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
#include <chrono>
#include <queue>


template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static float distance(const float &a, const float &b)
    {
        return std::abs(a - b);
    }
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


// KD-tree node
template <typename T>
struct Node
{
    T embedding;
    // std::string url;
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
    int start,
    int end,
    int depth
) {
    // if (depth == 2) return nullptr;
    if (start == end) return nullptr;
    int d = Embedding_T<T>::Dim();
    int split_dim = depth % Embedding_T<T>::Dim();

    std::sort(items.begin() + start, items.begin() + end, [split_dim, d](const std::pair<T,int>& a, const std::pair<T,int>& b) {
        for (int i=split_dim; i < d; i++) {
            float diff = getCoordinate(a.first, i) - getCoordinate(b.first, i);
            if (diff == 0) continue;
            return diff < 0;
        }
        for (int i=0; i < split_dim; i++) {
            float diff = getCoordinate(a.first, i) - getCoordinate(b.first, i);
            if (diff == 0) continue;
            return diff < 0;
        }
        return getCoordinate(a.first, split_dim) < getCoordinate(b.first, split_dim);
    });

    // for (int i=0; i<(int)items.size(); i++) {

    //     if constexpr (std::is_same_v<T, float>) {
    //         std::cout << items[i].first << ", ";
    //     } else {
    //         for (auto j : items[i].first) std::cout << j << ' ';
    //         std::cout << ", ";
    //     }
    // }
    // std::cout << std::endl;
    int mid = (start + end - 1) / 2;
    // if ((start + end) % 2 == 0) {
    //     int a = (start + end - 1) / 2;
    //     int b = (start + end) / 2;

    //     if (items[a].second < items[b].second) {
    //         mid = a;
    //     }
    //     else mid = b;
    // }
    Node<T>* root = new Node<T>();
    // if (depth ==0 ) {    if constexpr (std::is_same_v<T, float>) {
    //         std::cout << items[mid].first << std::endl;
    //     } else {
    //         for (auto i : items[mid].first) std::cout << i << std::endl;
    //     }}
    //     if (depth ==0 ) {    if constexpr (std::is_same_v<T, float>) {
    //         std::cout << items[mid + 1].first << std::endl;
    //     } else {
    //         for (auto i : items[mid + 1].first) std::cout << i << std::endl;
    //     }}
    root->embedding = items[mid].first;
    root->idx = items[mid].second;
    // std::cout << std::distance(items.begin(), begin) << ' ' << std::distance(items.begin(), end) << std::endl;
    // if (depth == 0) std::cout << start << " " << end << std::endl;

    root->left = buildKD_aux(items, start, mid, depth + 1);
    root->right = buildKD_aux(items, mid + 1, end, depth + 1);

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
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    return buildKD_aux(items, 0, (int) items.size(), depth);
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
    /*
    TODO: Implement this function to perform k-nearest neighbors (k-NN) search on the KD-tree.
    You should recursively traverse the tree and maintain a max-heap of the K closest points found so far.
    For now, this is a stub that does nothing.
    */
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
    float hdist = heap.top().first;
    if (std::abs(getCoordinate<T>(Node<T>::queryEmbedding, split_dim) - getCoordinate<T>(node->embedding, split_dim)) < hdist || (int) heap.size() < K) {
        knnSearch(far, depth + 1, K, heap);
    }
    return;
}