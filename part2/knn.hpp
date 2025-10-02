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
// #include <omp.h>
// 4, 5, 12920, 26743, 6, 0, 3, 68740, 40982, 63176,
#define N 4

#if defined(__SSE2__)
    #include <emmintrin.h> // Header for SSE2
#endif

// KD-tree node
template <typename T>
struct SubNode {
    // Should be precisely 32 bytes, since int+short+short+T is 4+2+2+24  NVM
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
template <typename T> struct ThreadPool;
template <typename T> inline ThreadPool<T>& thread_pool();

template <typename T>
struct ThreadTask {
    //  janky way to have threads persist, so hopefully build and search
    //  use the same cores for data subsection.
    public:
        std::condition_variable wakeup;

        void push(std::function<void()> t) {
            // std::cout << "New task pushed" << std::endl;
            std::lock_guard<std::mutex> lock(mutex);
            if (task) {
                std::cout << "New task assigned while old still in effect" << std::endl;
                return;
            }
            task = t;
            wakeup.notify_all();
        }

        void doit() {
            // std::cout << "Task tried" << std::endl;
            std::lock_guard<std::mutex> lock(mutex);
            if (!task) {
                // Spurious wakeups mean it can happen
                std::cout << "Task tried when none exists" << std::endl;
                return;
            }

            task();
            task = nullptr;
            thread_pool<T>().dec_active();
        }

        bool hasTask() {
            if (task) return true;
            return false;
        }

    private:
        std::mutex mutex;
        std::function<void()> task;
};

template <typename T>
struct ThreadPool {
    public:
        ThreadPool() {
            for (size_t i = 0; i < N; ++i) {
                threads[i] = std::thread([this, i] {
                    this->thread_func(i);
                });
            }
        }

        ~ThreadPool() {
            stop = true;

            for (int i=0; i<N; i++) {
                tasks[i].wakeup.notify_all();
            }

            for (std::thread &thread : threads) {
                thread.join();
            }
        }

        void dec_active() {
            std::lock_guard lock(mtx);
            num_active--;
            if (num_active == 0) all_finished.notify_all();
        }

        void submit_task( int thread_num, std::function<void()> task ) {
            // Don't lock here
            {
                std::lock_guard lock(mtx);
                num_active++;
            }
            tasks[thread_num].push(task);
        }

        void thread_func(int i);

        void wait() {
            // std::cout << "Waiting for all to finish" << std::endl;
            if (num_active == 0) return;
            while (true) {
                std::unique_lock lock(mtx);
                all_finished.wait(lock, [this] { return this->num_active == 0; });

                if (num_active == 0) return;
            }
        }

    
    private:
        std::condition_variable all_finished;
        std::mutex mtx;
        int num_active = 0;
        std::array<ThreadTask<T>, N> tasks;
        std::array<std::thread, N> threads;
        bool stop = false;
};

template<typename T>
void ThreadPool<T>::thread_func(int i) {
    while (true) {
        {
            std::unique_lock<std::mutex> lock(mtx);
            tasks[i].wakeup.wait(lock, [this, i] { return stop || tasks[i].hasTask(); });

            if (stop) return;
        }

        tasks[i].doit();
    }
}

template <typename T>
inline ThreadPool<T>& thread_pool() {
    static ThreadPool<T> pool = ThreadPool<T>();
    return pool;
}

template <typename T, typename = void>
struct Embedding_T;

// scalar float: 1-D
template <>
struct Embedding_T<float>
{
    static size_t Dim() { return 1; }

    static constexpr float distance(const float &a, const float &b) { return std::abs(a - b); }
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
    
    static float distance_aux (
        const float* __restrict__ a,
        const float* __restrict__ b,
        const size_t dim
    ) {
    #if defined(__SSE2__)
        __m128 sum_vec = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < dim; i += 4) {
            __m128 vec_a = _mm_loadu_ps(a + i);
            __m128 vec_b = _mm_loadu_ps(b + i);
            __m128 diff = _mm_sub_ps(vec_a, vec_b);
            __m128 sq = _mm_mul_ps(diff, diff);
            sum_vec = _mm_add_ps(sum_vec, sq);
        }
        float buffer[4];
        _mm_storeu_ps(buffer, sum_vec);
        float s = buffer[0] + buffer[1] + buffer[2] + buffer[3];
        for (; i < dim; ++i) {
            float d = a[i] - b[i];
            s += d * d;
        }
        return std::sqrt(s);

    #else
        float s = 0;
        #pragma omp simd reduction(+:s)
        for (size_t i = 0; i < dim; ++i) {
            float d = a[i] - b[i];
            s += d * d;
        }
        // return s;
        return std::sqrt(s);
    #endif
    }
                            

    static float distance(const std::vector<float> &a,
                          const std::vector<float> &b)
    {
        return distance_aux(a.data(), b.data(), Dim());
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

// Definition of static member
template <typename T>
T Node<T>::queryEmbedding;

template <typename T>
void buildKD_aux(
    std::vector<std::pair<T,int>>& items,
    Node<T>* root,
    int start,
    int end,
    int depth,
    int thread_num = -1
) {
    if (start == end) return;
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

    if (depth == 0) {
        thread_pool<T>().submit_task(1, [&items, root, mid, end, depth]() {
            return buildKD_aux(items, root, mid + 1, end, depth + 1, 1);
        });
        buildKD_aux(items, root, start, mid, depth + 1);
    } else if (depth == 1) {
        // std::cout << thread_num << std::endl;
        thread_pool<T>().submit_task(thread_num + 1, [&items, root, mid, end, depth, thread_num]() {
            return buildKD_aux(items, root, mid + 1, end, depth + 1, thread_num + 1);
        });
        buildKD_aux(items, root, start, mid, depth + 1);
    } else {
        buildKD_aux(items, root, start, mid, depth + 1);
        buildKD_aux(items, root, mid + 1, end, depth + 1);
    }
    curr->left_idx = (start == mid) ? -1 : (start + mid) / 2;
    curr->right_idx= (mid + 1 == end) ? -1 : (mid + 1 + end) / 2;

    return;
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
    thread_pool<T>().wait();
    // std::cout << 1;
    return root;
}


template <typename T>
void freeTree(Node<T> *node) {
    if (!node) return;
    delete node;
}


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
    if (depth > 1) {
        if (close != 65535) {
            knnSearch_aux(root, close, depth + 1, K, heap);
        }
        if ((int) heap.size() < K || std::abs(getDiffAtCoor<T>(*(node->embedding), Node<T>::queryEmbedding, split_dim)) < heap.top().first) {
            if (far != 65535) knnSearch_aux(root, far, depth + 1, K, heap);
        }
    }
    else {
        MaxHeap close_heap;
        auto fleft = std::async(std::launch::async, [root, close, depth, K, &close_heap]() {
            knnSearch_aux(root, close, depth + 1, K, close_heap);
        });

        if ((int) heap.size() < K || std::abs(getDiffAtCoor<T>(*(node->embedding), Node<T>::queryEmbedding, split_dim)) < heap.top().first) {
            if (far != 65535) knnSearch_aux(root, far, depth + 1, K, heap);
        }
        fleft.get();
        while (!close_heap.empty()) {
            heap.push(close_heap.top());
            close_heap.pop();
        }
        while ((int)heap.size() > K) {
            heap.pop();
        }
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

    // std::cout << 2;
    knnSearch_aux(root, root->mid, depth, K, heap);
}