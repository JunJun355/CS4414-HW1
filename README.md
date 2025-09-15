# Overview
The goal of this assignment is to give you a practical understanding of how K-Nearest Neighbors (KNN) search works, starting from simple 1D data and progressing to high-dimensional, real-world applications. You will begin by building a basic k-d tree and implementing a KNN search algorithm from scratch. Then, you'll extend your solution to work with multi-dimensional data, like text embeddings used in semantic search. In the final part, you'll use the ALGLIB library to explore how optimized and approximate KNN search can improve performance.


By the end of the assignment, you should be able to:
- Build a balanced k-d tree from both 1D and multi-dimensional data.
- Implement a recursive KNN search that efficiently finds the closest points to a query.
- Work with external libraries (ALGLIB) to perform KNN and compare exact vs. approximate search methods.
- Understand practical trade-offs between accuracy and speed in real-world search systems.

Sections II-III explain the entire overview of this assignment. We recommend that you read and understand these first before proceeding to the implementations in Sections IV-VI.


The assignment is broken down into three main parts with three different deadlines:

| Task                                         | Points | Due Date             |
|----------------------------------------------|--------|----------------------|
| 1. Work with 1D Data                         | 40     | September 15, 2025   |
| 2. Work with k-d Data                        | 20     | September 22, 2025   |
| 3. ANN Search Using the ALGLIB Library       | 30     | September 29, 2025   |

For each task, we have provided you with starter code and unit tests. You need to submit the
code to Gradescope with your implementation at each due date.

# I. Background

Many problems in computing involve K-Nearest Neighbor (KNN) Search—identifying the K points in a dataset that are closest to a given query point. This primitive underlies a wide range of applications, including recommendation systems (e.g., finding similar users or items) and natural language processing (e.g., retrieving semantically similar words or sentences).

In these applications, each data point—such as a user profile, product, or sentence—is typically represented by a k-dimensional embedding. These embeddings are constructed so that semantically similar items are mapped to points that lie close together in the high-dimensional space; in other words, the closer two points are in embedding space, the more similar they are in meaning or behavior.

To efficiently perform KNN search over large collections of embeddings, we use spatial indexing structures like the k-d tree. The k-d tree recursively partitions the space by axis-aligned splits, enabling fast pruning of regions that are far from the query point and significantly reducing the number of distance computations required.

# II. KNN Search with 1D Data

To illustrate the core ideas behind KNN search on a k-d tree, let’s begin with a simple case: 1-dimensional (k=1) data. Suppose we are given a list of scalars, [1, 3, 4.2, 5, 6, 7.8, 9], and our goal is to answer queries like: “What are the two numbers closest to 4?”

## Building the 1-d Tree

In one dimension, a k-d tree reduces to a binary search tree (BST). To enable efficient nearest-neighbor queries, we organize the points by recursively splitting the search space at the median:

A good strategy is to split the data using the **median** so that we can have a balanced tree:

1. Sort the data: `[1, 3, 4.2, 5, 6, 7.8, 9]`
2. Use the median `5` as the root.
3. Recursively:
   - Left of 5: build subtree from `[1, 3, 4.2]`
   - Right of 5: build subtree from `[6, 7.8, 9]`

This yields the following tree:

          5
        /   \
      3      7.8
     / \    /   \
    1  4.2 6     9

### Why Median Splits Matter?

Splitting at the median ensures that the tree is balanced, meaning the height of the tree is logarithmic in the number of points. This balance is crucial for search efficiency:
- With a balanced tree, the query time is O(log n) in the best case.
- An unbalanced tree (e.g., one built by inserting points in sorted order) can degenerate into a linked list, resulting in linear-time queries.

### Splitting the Search Space

Each node in the tree partitions the 1D space into two regions:

- Values less than or equal to the node go to the left subtree.
- Values greater than the node go to the right subtree.



## KNN Search on 1-d tree

Once we construct a k-d tree (which reduces to a binary search tree in 1D), we can efficiently find the k nearest neighbors of a query point by recursively searching the tree and pruning any subtree that cannot possibly contain a closer point.

### Initialization
Maintain a max-heap (or priority queue) of size at most k, where each entry stores a point and its distance to the query.
The heap keeps track of the k closest candidates found so far.
The top of the heap always contains the farthest point among the current top-k.

### Recursive Search Procedure

To perform the search, we apply the following recursive procedure to each node, starting from the root of the tree.

At each node:

- Step 1: Process the Current Node

  - Compute the distance d between the query point and the current node’s value.
  - Update the max-heap:
    - If the heap contains fewer than k points, insert (d, value) directly.
    - If the heap is full and d is smaller than the largest distance in the heap, replace the farthest point.

- Step 2: Recursively Search the Near Subtree

  - Determine which subtree is "near":
    - If query < node.value, the near subtree is the left child.
    - Otherwise, it's the right child.
  - Recursively apply this same procedure (step 1-3) to the near child: at each recursive call, continue updating the heap using the same logic.

- Step 3: Decide Whether to Search the Far Subtree
  - If the heap contains fewer than k points, we must search the far subtree to ensure we find k total neighbors.
  - Otherwise, use the following logic to decide:
    - Let d_split = |query - node.value|, the distance from the query to the current node’s split point. Let d_max be the largest distance currently in the heap.
    - If d_split < d_max, then the far subtree might contain a closer point. This is because the node’s value partitions the 1D space, and every point in the far subtree lies at least d_split away from the query. So if d_split < d_max, some points in the far subtree could be closer than the current farthest neighbor → recurse into it.
    - If d_split ≥ d_max, then no point in the far subtree can be closer than what we already have → prune the subtree.

---

Let’s walk through how the algorithm finds the 2 nearest neighbors to the query 4.9, using the 1D k-d tree constructed earlier.

- Visit node 5
  - Distance = |5 - 4.9| = 0.1
  - Add to heap → best = [(5, 0.1)]
  - Since 4.9 < 5, go to the left child (3)
  ---
  - Visit node 3
    - Distance = |3 - 4.9| = 1.9
    - Add to heap → best = [(5, 0.1), (3, 1.9)]
    - Since 4.9 > 3, go to the right child (4.2)
    ---
    - Visit node 4.2
      - Distance = |4.2 - 4.9| = 0.7
      - Heap is full. Current worst = (3, 1.9)
      - Replace 3 with 4.2 → best = [(5, 0.1), (4.2, 0.7)]
      - Both children are null → return to node 3
      ---
  - Back at node 3 (after right child)
    - Check whether to visit the left child
    - Distance to split = |4.9 - 3| = 1.9
    - Worst in heap = 0.7
    - Since 1.9 > 0.7, we do not need to explore the left child → prune

- Back at node 5 (after left child)
  - Check whether to visit the right child
  - Distance to split = |4.9 - 5| = 0.1
  - Worst in heap = 0.7
  - Since 0.1 < 0.7, we must explore the right child (7.8)
  ---
  - Visit node 7.8
    - Distance = |7.8 - 4.9| = 2.9 → too far → no update
    - Since 4.9 < 7.8, go to the left child (6)

    ---
    - Visit node 6
      - Distance = |6 - 4.9| = 1.1 → worse than current worst (0.7) → no update
      - Both children are null → return to node 7.8

  - Back at node 7.8 (after left child)
    - Check whether to visit the right child
    - Distance to split = |4.9 - 7.8| = 2.9
    - Worst in heap = 0.7
    - Since 2.9 > 0.7, we do not need to explore the right child → prune

The 2 nearest neighbors to 4.9 are: [5, 4.2]


# III. KNN Search with k-d Data

So far, we’ve seen how to build a k-d tree for 1D data by recursively splitting the data at the median to form a balanced binary search tree. Now we generalize this idea to handle data points in $\mathbb{R}^d$.

A k-d tree organizes $k$-dimensional points using recursive partitioning. At each level of the tree, we split the space along one of the $k$ dimensions, rotating through dimensions as we go deeper in the tree. Each node effectively places a hyperplane (e.g., a vertical or horizontal line in 2D) that divides the space into two halves.

## Build k-d Tree

A **k-d tree** (short for *k*-dimensional tree) is a generalization of a binary search tree that organizes points in $k$-dimensional space. Like the 1D case, the idea is to recursively partition the space, but instead of splitting the number line, each node splits the space with a **hyperplane** (e.g., a vertical or horizontal line in 2D). This recursively divides the space into regions, like slicing a cake repeatedly along different planes.

### Key Differences from 1D

- In 1D, we always split along a single axis (the scalar value). In k-d, we **cycle through the $k$ dimensions** as we go deeper in the tree. For example, in 2D, it goes `x → y → x → y → ...`.
- At each node, we:
  - Choose the median value in the current dimension.
  - Partition the remaining points into a **left subtree** (values below the median in the current dimension) and a **right subtree** (values above).

Suppose we are given a set of 2d points: `[(3, 6), (17, 15), (13, 15), (6, 12), (9, 1), (2, 7), (10, 19)]`. Here is how we build a k-d tree:


1. Sort by x-coordinate (dimension 0): `[(2, 7), (3, 6), (6, 12), (9, 1), (10, 19), (13, 15), (17, 15)]`
2. Use the median `(9, 1)` as the root.
3. Recursively:
   - Left: build subtree from `[(2, 7), (3, 6), (6, 12)]` (sort by y-coordinate (dimension 1))
   - Right: build subtree from `[(10, 19), (13, 15), (17, 15)]` (sort by y-coordinate (dimension 1))

<!---
- Sort by x-coordinate (dimension 0): [(2, 7), (3, 6), (6, 12), (9, 1), (10, 19), (13, 15), (17, 15)]
- Use the median (9, 1) as the root
- Recursively build the left subtree: [(2, 7), (3, 6), (6, 12)]
  - Sort by y-coordinate (dimension 1): [(3, 6), (2, 7), (6, 12)]
  - Use the median (2, 7) as the root
    - Left child: (3, 6)
    - Right child: (6, 12)
- Recursively build the right subtree: [(10, 19), (13, 15), (17, 15)]
  - Sort by y-coordinate (dimension 0): [(13, 15), (17, 15), (10, 19)]
  - Use the median (17, 15) as the root
    - Left child: (13, 15)
    - Right child: (10, 19)
--->

This yields the following tree:


                       (9, 1)                     ← level 0, split on x
                      /      \
            (2, 7)                 (17, 15)       ← level 1, split on y
           /     \                /       \
      (3, 6)   (6, 12)       (13, 15)   (10, 19)  ← level 2, split on x

---

## KNN Search on k-d Tree

Now that we understand how to build a k-d tree, we can extend the KNN search procedure from 1d to k-d data. Most of the algorithm remains conceptually the same: we maintain a max-heap of the best k candidates and recursively search the tree. However, there are some key differences:


- Instead of computing absolute differences between scalars, we compute Euclidean distances between vectors to determine how close a point is to the query.

- Each node splits the space using a hyperplane perpendicular to one of the coordinate axes. The axis used at each level is called the splitting dimension, which cycles through dimensions:
  - At depth d, we use axis = d % k.

- To decide which subtree is the near tree, we only compare the query and current node along the splitting dimension:
  - If query[axis] < node[axis] → search left child first
  - Else → search right child first


- To decide whether we need to explore the far subtree, we compute the distance from the query to the splitting hyperplane:
  - In k-d, this is simply |query[axis] - node[axis]|
  - If this value is smaller than the current worst distance in the heap, the far subtree might contain a closer point and must be explored.



# IV. Part 1: Work with 1D Data (40 points, due in 2 weeks)
You are required to implement two logics:
- Building a k-d tree
- Performing KNN search over the k-d tree

Your implementation in this part only needs to handle 1-dimensional data.

## Starter Code
You will work with the following three files in the `part1` directory:
- `main.cpp`: This is the entry point of the program. It handles command-line arguments, loads and parses the query and passages JSON files, extracts embeddings, builds a KD-tree, performs k-nearest neighbors (KNN) search, and prints the results and performance metrics. It only supports scalar float embeddings (mode 0).

- `knn.hpp`: This header file defines the types and function prototypes used for the KD-tree and KNN search. It includes the Node struct for KD-tree nodes, type aliases for embeddings and priority queues, and declarations for functions like buildKD and knnSearch.

- `knn.cpp`: This file implements the functions defined in `knn.hpp`.


A Makefile is provided in the part1 directory to simplify the compilation of the program. This allows you to build the project and its dependencies with a single command. To **compile** the program:

```bash
cd part1
make all
```

To run the program:

```bash
./main <query_json> <passage_json> <K>
```

- `<query_json>`: Path to the JSON file containing the query point.

- `<passage_json>`: Path to the JSON file containing the candidate data points. The program will search for neighbors among these elements.

- `<K>`: Number of nearest neighbors to find


## Data Structures and Functions
### `Embedding_T`

`Embedding_T` is defined as an alias for float, representing a scalar (1-dimensional) embedding:
```
using Embedding_T = float
```

### `Node`

The `Node` structure represents a node in the K-D tree. This structure enables recursive construction and traversal of the K-D tree, and will be the building block for implementing the tree construction and KNN search.

- `embedding`: the point in *k*-dimensional  (k=1 here) space (of type `Embedding_T`).
- `idx`: the index of the point in the dataset.
- `left` and `right`: pointers to the left and right child nodes.
- `queryEmbedding`: a static member used during nearest neighbor search to compare against a fixed query point.


## Your jobs

### A. Implement the `buildKD` function (20 points)
You are required to implement the `buildKD` function, which recursively constructs a **balanced K-D tree** from the data points stored in the vector `allPoints`.

Arguments:
- `items`: A vector of (embedding, id) pairs representing the data points to be organized in the current (sub)tree.
- `depth`: The current depth in the K-D tree recursion, used to determine the splitting axis. Defaults to 0.

At each recursive step:

- Determine the **splitting axis** by cycling through dimensions. Note that in part 1, there is only one dimension for 1-dimensional data, so this step has no effect.
- Sort the current subset of points based on the selected axis.
- Select the **median point** (by index) to serve as the root of the current subtree. This ensures the tree remains balanced.
- Recursively apply `buildKD` to the data points on the left and right of the median to construct the left and right subtrees.


Returns:
- A pointer to the root node of the constructed (sub)tree.


#### Grading and Output Validation

To help us validate your implementation, we have provided a utility function printTree that traverses the k-d tree using depth-first search (DFS) and prints each node during traversal. In the runMain() function, we call printTree() on your constructed tree. You will be graded based on whether the output of printTree  matches the expected output on the provided test cases.

When multiple data points share the same value along the splitting dimension, there may be more than one valid median. To guarantee that your tree matches the expected output, you must follow the rules below:

- Sorting rule:
  - At each level of the tree, determine the current splitting dimension:
  axis = depth % d, where d is the number of dimensions.
  - Sort the current list of points using the following comparison logic:
    - First, compare points based on their value in the current splitting dimension.
    - If two points are equal along that dimension, compare their values in the next dimension: (axis + 1) % d.
    - If still tied, continue comparing values in subsequent dimensions, cycling through all d dimensions in order, until the tie is broken.
  - You can assume there will never be a complete tie (no two points are exactly the same in every dimension).

- Median selection rule:
  - After sorting the list of points as described above, select the median point by index.
    - If the number of points is odd, select the middle point as usual.
    - If the number of points is even, select the first of the two middle points in the sorted list (i.e., the one with the lower index).


We have generated four synthetic datasets, 1d-1.json, 1d-5.json, 1d-10.json, and 1d-100.json, located in the ./data/ directory, containing 1, 5, 10, and 100 data points respectively. We will run the program with each of these 4 datasets and verify the correctness of the built tree. Each case contributes 5 points.


### B. Implement the `knnSearch` function (20 points)

You are also required to implement the `knnSearch` function, which recursively traverses the k-d tree to find the k nearest neighbors of a given query point. The value of `K` (the number of nearest neighbors to find) will always be smaller than or equal to the total number of points in the tree. In other words, there will always be enough data points to satisfy the request.

Arguments:

- `node`: A pointer to the current node in the K-D tree.
- `depth`: The current recursion depth, used to determine the splitting axis.
- `K`: The number of nearest neighbors to find.
- `heap`: A max-heap that stores the current best K candidates, ordered by distance from the query point.


At each recursive step:

- Use the current `depth` to compute the **splitting axis**:  
  `axis = depth % Embedding_T<T>::Dim`.
- Compare the query point (`Node<T>::queryEmbedding`) to the current node’s point along the splitting axis.
- Based on this comparison, **recursively explore the subtree** that is closer to the query point (left or right).
- After visiting the near subtree:
  - Compute the distance between the query point and the current node.
  - Add the node to the max-heap `heap` if it is among the current best `K` candidates.
- Then, **conditionally explore the far subtree**. You only need to recurse into it if the hyperplane distance along the splitting axis is smaller than the current farthest distance in the heap — this is the core idea that makes K-D tree search efficient.

Returns:
- This function does not return a value. Instead, it updates the heap in-place with the top-K nearest neighbors discovered so far.


#### Grading and Output Validation

To help us validate your implementation, we have provided a utility function printNeighbour, which prints the elements stored in the heap in ascending order of distance from the query point. The runMain() function will call printNeighbour() after your knnSearch function completes. You will be graded based on whether the output of printNeighbour matches the expected output on the following test cases:

- Find 1 nearest neighbour among 1 data points (2.5 points)
- Find 1 nearest neighbour among 5 data points (2.5 points)
- Find 2 nearest neighbours among 5 data points (2.5 points)
- Find 1 nearest neighbour among 10 data points (2.5 points)
- Find 2 nearest neighbours among 10 data points (2.5 points)
- Find 1 nearest neighbour among 100 data points (2.5 points)
- Find 5 nearest neighbours among 100 data points (2.5 points)
- Find 10 nearest neighbours among 100 data points (2.5 points)













# V. Part 2: Work with k-d Data (20 points, due in 1 week)

In this part, you will generalize your 1D implementation to handle multi-dimensional data. Real-world problems often require reasoning over high-dimensional inputs—for example, text embeddings, image features, or sensor data. You will reuse the same core logic from Part 1 (reading data, building the tree, and running KNN search), but extend it to operate over vectors in $\mathbb{R}^d$ instead of scalars.


## Real-World Motivation: Semantic Search for Documents:

Imagine you’re building a search engine for a large collection of documents—thousands of articles, essays, or support tickets. A user types in a query like: “Best ways to reduce GPU memory usage”. You want to return documents that are semantically similar to this question. Traditional keyword-based methods (like grep or inverted indices) search for exact word matches, but they miss documents that use different phrasing—e.g., “Techniques for VRAM optimization” and “How to lower GPU memory consumption”. Even though these mean similar things, a keyword search would miss them. To solve this, we use semantic search, which compares documents based on their meaning rather than literal word overlap.


## Embeddings: turning text into points in high-dimensional space

To compare meaning, we first map each document (and each query) into a high-dimensional vector, a list of numbers, in such a way that semantically similar texts end up close together in that space.

Embedding generation: we leverage an ML model which maps text to a k-dimensional vector in $\mathbb{R}^k$ (e.g. a list of 128, 512, 768, 1024 ... numbers, in our case, the dimension is 384). There exist several common models:

- **Word2Vec / GloVe**: Shallow models that learn word vectors by predicting surrounding context words.
- **BERT / Sentence-BERT**: Deep transformer models that produce context-aware token embeddings; you can average or use a special token embedding to get a single vector for a full sentence or document.

In this homework, we use the BGE-small-en-v1.5 model from Hugging Face to embed each passage into a 384-dimensional vector. These embeddings are constructed so that semantically similar items are mapped to points that lie close together in the high-dimensional space; in other words, the closer two points are in embedding space, the more similar they are in meaning or behavior.

## Dataset: MS MARCO (Passage Ranking)
We use the MS MARCO passage ranking dataset (Li et al., arXiv:1611.09268), a benchmark commonly used in semantic search and retrieval tasks. You can learn more about it from the [MS MARCO website](https://microsoft.github.io/msmarco/).

We’ve processed 9650 queries and 79,176 passages from the dataset by embedding each one using the BGE encoder. They are split into two smaller files `passages1.json` and `passages2.json`.

To download the data:

```
chmod +x download.sh
./download.sh
```

The queries are stored in (the first one is served as the query point in the program):

```
./data/queries_emd.json
```

An example of one of the passage files is stored in:

```
./data/passages1.json
```

To test your program with this dataset, run:
```
./main 384 ./data/queries_emd.json ./data/passages1.json 2
```

This command runs KNN search in 384D to find the 2 nearest neighbors to the query point.

We have also uploaded several simple test cases (e.g., `3d-1.json`) in `part2/data` that you can use for debugging.



## Starter Code
You will work with the following two files in the `part2` directory:
- `main.cpp`: This is the entry point of the program, similar to the one used in part 1.

- `knn.hpp`: This header file defines the types and function as well as the implementation used for the KD-tree and KNN search. In Part 2, we use a template class for `T`, which can represent either a `float` (for 1-dimensional scalar data) or an array of floats (for k-dimensional data). This design allows the same code to handle both 1D and multi-dimensional cases in a generic way. In C++, template implementations must be placed in header files (`.hpp`) because the compiler needs to see the full implementation when instantiating templates. Therefore, for this project, all implementation is provided in the `.hpp` files, and `.cpp` source files are not used in Part 2.


To **compile** the program:

```bash
cd part2
make all
```

To run the program:

```bash
./main <dim> <query_json> <passage_json> <K>
```

- `<dim>`:  
  - The dimensionality of the embedding.
  - The embeddings generated using the described process have dim = 384. However, you may also create toy embeddings with smaller dimensions for debugging purposes.

- `<query_json>`: Path to the JSON file containing the query point.

- `<passage_json>`: Path to the JSON file containing the candidate data points. The program will search for neighbors among these elements.

- `<K>`: Number of nearest neighbors to find


### `Embedding_T<T>`

The `Embedding_T<T>` structure uses C++ template specialization to allow the K-D tree implementation to work with both scalar and vector embeddings in a generic and extensible way.

- For `float`: 
  - Represents a scalar (1-dimensional) embedding.
  - The `Dim` function always return 1.
  - The `distance` function computes the absolute difference: `|a - b|`.

- For `std::vector<float>`:
  - Represents a fixed-size N-dimensional vector embedding.
  - The `Dim` function returns N, which corresponds to the dim value of the argument you provided.
  - The `distance` function computes the Euclidean distance between two vectors.

### `Node<T>`

The `Node<T>` structure represents a node in the K-D tree. This structure enables recursive construction and traversal of the K-D tree, and will be the building block for implementing the tree construction and KNN search.

- `embedding`: the point in *k*-dimensional space (of type `T`).
- `idx`: the index of the point in the dataset.
- `left` and `right`: pointers to the left and right child nodes.
- `queryEmbedding`: a static member used during nearest neighbor search to compare against a fixed query point.


### `PQItem`
This is a type alias for a `std::pair<float, int>`. It is used to represent an item in a priority queue, where the float value indicates the priority (the distance between an embedding and a query embedding), and the int value is an associated index for the embedding.


### `MaxHeap`
This is a type alias for a max-heap priority queue of PQItem elements. It uses `std::priority_queue` with PQItem as the value type, `std::vector<PQItem>` as the underlying container, and `std::less<PQItem>` as the comparison function. This configuration ensures that the item with the highest priority (largest float value) is always at the top of the heap.


## Your Job
You will reuse the same core logic from Part 1 (building the tree, and running KNN search), but extend the implmentation of function `buildKD` and `knnSearch` to operate over k-d embedding, either vectors in $\mathbb{R}^d$ and scalars.


### Grading
You will be tested on four different values of k: 1, 3, 5, and 10.  There are four test cases each in k = 1, 3, and 5. The last test case of these are based on the MS MARCO passage datasets; the other datasets are located in the `./data/` directory. There are two test cases in k = 10, both based on the MS MARCO passage datasets.

We will run your program to verify the correctness of the built tree and the output of the search. Each value of `k` total to be worth 5 points. The `buildKD` on the MS MARCO dataset 1 and 2 across all `k` are combined into the `buildKD, MS MARCO 1` and `buildKD, MS MARCO 2` tests.

- Correct outputs for k=1 → 5 points
   - 1.25 points each test case (4 test cases)
   - Each test case split into `buildKD` and `knnSearch`
- Correct outputs for k=3 → 5 points
   - 1.25 points each test case (4 test cases)
   - Each test case split into `buildKD` and `knnSearch`
- Correct outputs for k=5 → 5 points
   - 1.25 points each test case (4 test cases)
   - Each test case split into `buildKD` and `knnSearch`
- Correct outputs for k=10 → 5 points
   - 2.5 points each test case (2 test cases)
   - Each test case split into `buildKD` and `knnSearch`



# VI. Part 3: ANN Search Using the ALGLIB Library (30 points, 1 week)


In this part, your goal is to reproduce the KNN functionality you implemented in Part 1 and Part 2, but now using the ALGLIB library.

- In Part 1, you implemented KNN search for 1D data using a binary tree.
- In Part 2, you extended that to perform KNN search for multidimensional data.
- In Part 3, you will use ALGLIB’s built-in k-d tree and KNN search routines to reproduce the same functionality, and compare the behavior of ALGLIB’s implementation to your own. ALGLIB’s KNN search is built on a balanced k-d tree, similar to the one you implemented earlier, but with optimized pruning techniques and support for approximate nearest neighbour (ANN) search using a parameter called epsilon (ε).



## ALGLIB Data Structures and APIs for KNN Search

 ALGLIB provides its own array wrappers and search routines for working with numerical data. This section explains the core data types and functions you'll need.

### Core Array Types

ALGLIB provides special array types that wrap native C++ arrays. These types are used for compatibility with ALGLIB’s internal memory management and numerical routines.

- `real_1d_array`
  - Represents a 1D array of floating-point values (double)
  - Used to store a single query point

  Example: 
  ```
  alglib::real_1d_array query;
  query.setlength(3);
  query[0] = 1.0;
  query[1] = 2.0;
  query[2] = 3.0;
  ```

- `real_2d_array`
  - Represents a 2D array of floating-point values (double)
  - Used to store the embeddings of all candidate data points
  - Each row is a data point
  - Each column is a dimension
  
  Example:
  ```
  double raw_data[] = {
      1.0, 2.0, 3.0,   // Point 0
      4.0, 5.0, 6.0    // Point 1
  };
  alglib::real_2d_array allPoints;
  points.setcontent(2, 3, raw_data);
  // Represents: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
  ```
f
- `integer_1d_array`

  - Represents a 1D array of integers
  - Each point in the input JSON file (e.g., the passages file) is a JSON object that contains the following fields: id, embedding, and text. You should extract the "embedding" and store it in a real_2d_array. You should also extract the "id" and store it in an integer_1d_arra. This is used to store the "id" tag of each data point (so we can retrieve the point's identity after search)

  Example:
  ```
  alglib::integer_1d_array tags;
  tags.setlength(2);
  tags[0] = 101;  // ID for point 0
  tags[1] = 102;  // ID for point 1
  ```

### ALGLIB key functions

Here are the key functions you'll use:

- `kdtreebuildtagged(...)`
  - This function builds a k-d tree from your dataset and attaches an integer tag (e.g., the 'idx' field of each data point) to each data point. This tag will be returned when querying the tree, allowing you to recover which original point was matched.
  - Arguments:
    - `points`: your dataset (real_2d_array), N rows × D columns
    - `tags`:  integer IDs for each point (integer_1d_array)
    - `N`: number of data points
    - `D`: number of dimensions
    - `NY`: number of optional Y-values, NY>=0 (we do not need to use Y-values here so just pass 0).
    - `normtype`: set to 2 for Euclidean distance
    - `tree`: the k-d tree object (alglib::kdtree tree) to build

  Example: Suppose: *allPoints* is a real_2d_array storing all embeddings (1 per row), and *tags* is an integer_1d_array storing the ID ("idx") of each point, you can build the tree like this:

  ```
  alglib::kdtree tree;
  alglib::kdtreebuildtagged(allPoints, tags, (int)N_points, (int)D, 0, 2, tree);
  ```

- `kdtreequeryaknn(...)`
  - This performs approximate KNN search.
  - Arguments:
    - `tree`: the built k-d tree
    - `query`: the query point (real_1d_array)
    - `K`: number of nearest neighbors to retrieve
    - `epsilon`: approximation factor (0 for exact search)
  - Return: the number of neighbors found. The return type is *ae_int_t*, which is Int type redefinition in alglib namespace.

  - Example:
  ```
  alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);
  ```


### kdtreequeryresultsdistances

This function retrieves the distances from the query point to each of the nearest neighbors found by the most recent query (e.g., after calling `kdtreequeryaknn`). The distances are returned in the same order as the tags/IDs from `kdtreequeryresultstags`.

- Arguments:
  - `tree`: the k-d tree object after a query
  - `distances`: a `real_1d_array` to be filled with the distances to the found neighbors

- Example:
  ```cpp
  alglib::real_1d_array dist;
  dist.setlength(k); // k = number of neighbors requested
  alglib::kdtreequeryresultsdistances(tree, dist);
  // dist[0], dist[1], ..., dist[k-1] now contain the distances to the nearest neighbors
  ```

  For example, after running a KNN query:
  ```cpp
  alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);
  alglib::real_1d_array dist;
  dist.setlength(count);
  alglib::kdtreequeryresultsdistances(tree, dist);
  for (int i = 0; i < count; ++i) {
      std::cout << "Neighbor " << i+1 << ": distance = " << dist[i] << std::endl;
  }
  ```


### kdtreequeryresultstags

This function retrieves the integer tags (IDs) of the nearest neighbors found by the most recent query (e.g., after calling `kdtreequeryaknn`). The tags correspond to the IDs you provided when building the tree with `kdtreebuildtagged`. This allows you to map the search results back to your original data points.

- Arguments:
  - `tree`: the k-d tree object after a query
  - `tags`: an `integer_1d_array` to be filled with the IDs of the found neighbors

- Example:
  ```cpp
  alglib::integer_1d_array idx;
  idx.setlength(k); // k = number of neighbors requested
  alglib::kdtreequeryresultstags(tree, idx);
  // idx[0], idx[1], ..., idx[k-1] now contain the IDs of the nearest neighbors
  ```

  For example, after running a KNN query:
  ```cpp
  alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, k, eps);
  alglib::integer_1d_array idx;
  idx.setlength(count);
  alglib::kdtreequeryresultstags(tree, idx);
  for (int i = 0; i < count; ++i) {
      std::cout << "Neighbor " << i+1 << ": id = " << idx[i] << std::endl;
  }
  ```

- 


## Starter code

You will work with the `main.cpp` in the `part3` directory.


To **compile** the program:

```bash
cd part3
Make all
```

To run the program:

```bash
./knn_alglib <query_json> <passage_json> <K> <eps>
```

- `<query_json>`: Path to the JSON file containing the query point.

- `<passage_json>`: Path to the JSON file containing the candidate data points. The program will search for neighbors among these elements.

- `<K>`: Number of nearest neighbors to find.

- `<epsilon>`: The approximation factor used in.


## Your job

### A. Implement ANN search using ALGLIB

You need to complete the knn_alglib.cpp file by:

1. Loading and Parsing the Input

    You should read the input JSON files for the query point and dataset (as in Part 1 and 2). Specifically, you will:

    - Load the "embedding" field from the query into a real_1d_array
    - Flatten all embeddings from the dataset into a double[] array and wrap it in a real_2d_array using setcontent(...)
    - Store all IDs (the 'idx' filed of each data point) into an integer_1d_array named as something like *tags*


2. Building the k-d Tree
Once you have wrapped the data into ALGLIB types, construct the k-d tree using:

    ```
    alglib::kdtreebuildtagged(allPoints, tags, (int)N_points, (int)D, 0, 2, tree);
    ```

3. Use kdtreequeryaknn(...) to perform the search:
    ```
    alglib::ae_int_t count = alglib::kdtreequeryaknn(tree, query, K, epsilon);
    ```


### B. Report

In addition to your implementation, you are required to submit a short report named report.pdf, located in the root directory of your homework submission.

Your should answer the following question in the report:

- How does your KNN Implementation compare with ALGLIB?
  - You should compare the performance of:
    - Your own k-d tree KNN implementation from Part 2
    - The ALGLIB-based implementation from Part 3
  - Run both implementations in exact mode (i.e., set <epsilon> = 0) and search for the 10 nearest neighbors (K = 10).
  - For each method, report the following timing breakdown:
    - Total elapsed time
    - Time to process and parse the input
    - Time to build the k-d tree
    - Time to perform the KNN search
  - You may use std::chrono or another timing library to measure these components. Present your results in a table for easy comparison.

- How does `epsilon` affect the speed and accuracy of ANN search?
  - ALGLIB supports approximate nearest neighbor (ANN) search using a tunable parameter called epsilon (ε). This parameter controls how aggressively the algorithm prunes parts of the k-d tree during search.

  - How epsilon affects pruning?
    - During search, ALGLIB keeps track of the distance to the best neighbors found so far. When epsilon > 0, the algorithm skips exploring branches that cannot improve the best result by more than a factor of (1 + ε). This allows it to avoid unnecessary computation, speeding up the search.
    - When ε = 0, the search is exact and will always return the true nearest neighbors.
    - When ε > 0, the search becomes faster but may return slightly less accurate results.
  - You should:
    - Eexperiment with different values of ε, such as: 0 (exact), 1, 2, 5, 10, 20
    - For each value, measure and report the time taken for the KNN search only (exclude data loading and tree construction).
    - Report your findings in the following table format:

        | ε (epsilon) | Search Time (ms) | Accuracy (out of 10) |
        | ----------- | ---------------- | -------------------- |
        | 0.0         | xx               | yy                   |
        | 0.5         | xx               | yy                   |
        | 1           | xx               | yy                   |
        | 2           | xx               | yy                   |
        | 5           | xx               | yy                   |
        | 10          | xx               | yy                   |

## Grading

Your work for Part 3 will be graded based on the correctness of your implementation and the quality of your written report. The total for this part is 20 points.

- Part 3.A — Implementation (15 points)
  - We will grade the correctness of your implementation (same as Part 2). To evaluate your results, we use the printNeighbour() function to print out the IDs and distances of the nearest neighbors your program finds. You will be tested on four different values of k: 1, 2, 3, 5, and 10.

  - Each of the five test cases is worth 3 points:
    - Correct output for k=1 → 3 points
    - Correct output for k=5 → 3 points
    - etc.

- Part 3.B — Report (15 points)

  - We will read your report (report.pdf) and evaluate it based on the following:
    - Clearly presenting timing comparisons between your implementation and ALGLIB for exact KNN search.
    - Including a table benchmarking the performance of approximate KNN search for different values of ε, as described in the instructions.
    - Offering a short discussion or insight into the trade-offs between speed and accuracy.

# VII. Submission Criteria

### Part 1.

Submit and upload the following files to the hw1 part 1 gradescope autograder:

- `knn.cpp`
- `knn.hpp`

### Part 2.

Submit and upload the following files to the hw1 part 2 gradescope autograder:

- `knn.hpp`

### Part 3.

Submit and upload the following files to the hw1 part 3 gradescope autograder:

- `knn_alglib.cpp`
- `report.pdf` (this will be manually reviewed)
