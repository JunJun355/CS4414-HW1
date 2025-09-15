#include "knn.hpp"
#include <vector>
#include <chrono>
#include <algorithm>

// Definition of static member
Embedding_T Node::queryEmbedding;


float distance(const Embedding_T &a, const Embedding_T &b)
{
    return std::abs(a - b);
}


constexpr float getCoordinate(Embedding_T e, size_t axis)
{
    return e;  // scalar case
}

Node* buildKD_assume_sorted(std::vector<std::pair<Embedding_T,int>>& items, int start, int end) {
    if (start == end) return nullptr;

    int mid = (start + end) / 2;
    Node* root = new Node();
    root->embedding = items[mid].first;
    root->idx = items[mid].second;
    root->left = buildKD_assume_sorted(items, start, mid);
    root->right = buildKD_assume_sorted(items, mid + 1, end);

    return root;

}

// Build a balanced KD‐tree by splitting on median at each level.
Node* buildKD(std::vector<std::pair<Embedding_T,int>>& items, int depth) {
    /*
    TODO: Implement this function to build a balanced KD-tree.
    You should recursively construct the tree and return the root node.
    For now, this is a stub that returns nullptr.
    */
    std::sort(items.begin(), items.end(), [](const std::pair<Embedding_T,int>& a, const std::pair<Embedding_T,int>& b) {
        return getCoordinate(a.first, 0) < getCoordinate(b.first, 0);
    });
    return buildKD_assume_sorted(items, 0, items.size());
}


void freeTree(Node *node) {
    if (!node) return;
    freeTree(node->left);
    freeTree(node->right);
    delete node;
}


void knnSearch(Node *node,
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

    float qdist = distance(node->embedding, Node::queryEmbedding);
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
    Node *close, *far;
    if (node->embedding > Node::queryEmbedding) {
        close = node->left;
        far = node->right;
    }
    else {
        close = node->right;
        far = node->left;
    }
    knnSearch(close, depth + 1, K, heap);
    float hdist = heap.top().first;
    if (qdist < hdist || (int) heap.size() < K) {
        knnSearch(far, depth + 1, K, heap);
    }
    return;
}