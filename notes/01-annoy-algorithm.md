# Annoy Algorithm Deep Dive

Source: `/home/sprite/annoy/src/annoylib.h` (~1597 lines, C++ templates)

## Core Idea

Annoy builds a **forest of random binary trees** that recursively partition the
vector space using random hyperplanes. At query time, it traverses all trees
simultaneously using a priority queue, collecting candidate items, then
computes exact distances for re-ranking.

## Data Structures

### Node Layout (Fixed Size)

Every node — leaf or internal — occupies the same fixed number of bytes:

```
_s = offsetof(Node, v) + f * sizeof(T)
```

```c++
struct Node {
    S n_descendants;       // How many items below this node
    union {
        S children[2];     // Child indices (or list of item IDs for small nodes)
        T norm;            // L2 norm of split vector (Minkowski metrics)
    };
    T v[f];                // Either: data vector (leaf), split hyperplane (internal),
                           //         or unused (small intermediate)
};
```

**Node types determined by `n_descendants`:**
- `n_descendants == 1` → **Leaf node**: `v` stores the actual data vector
- `2 <= n_descendants <= K` → **Small intermediate**: `children[]` array stores
  direct list of descendant item IDs (no split plane needed). K is calculated
  as `(node_size - offsetof(children)) / sizeof(S)`
- `n_descendants > K` → **Split node**: `v` stores the split hyperplane normal,
  `children[0]`/`children[1]` point to left/right child nodes

### On-Disk File Format

The file is a flat array of fixed-size nodes:

```
[Item 0][Item 1]...[Item N-1][Internal nodes...][Root copies...]
```

- File size = `_s * n_nodes`
- Items (leaves) are stored first at indices 0..n_items-1
- Internal nodes follow
- Root nodes are **copied** to the end of the file as a discovery mechanism
- On load, Annoy scans backwards from EOF to find roots (nodes where
  `n_descendants == n_items`)

### Multiple Trees

- Each tree is an independent binary tree over the same items
- Trees share the same leaf nodes (items at indices 0..n_items-1)
- Each tree has its own internal nodes and its own root
- Root node IDs are collected and stored at the end of the file

## Build Process

### Phase 1: Add Items
- `add_item(id, vector)` stores items sequentially in the node buffer
- Buffer grows with 1.3x realloc factor
- Can build on-disk (`on_disk_build()`) for datasets larger than RAM

### Phase 2: Build Trees (`build(n_trees, n_jobs)`)

For each tree, recursively partition items:

1. **Base case**: If subset has <= K items, create a "small" node storing item
   IDs directly in the `children[]` array
2. **Recursive case**:
   a. Run **two-means clustering** to find a split hyperplane:
      - Pick 2 random points as initial centroids
      - Run 200 iterations of weighted assignment
      - Split vector = centroid_a - centroid_b (normalized for angular)
   b. Classify items by `dot(item, split_vector)` ≷ 0
   c. **Rebalancing**: If split is >95% imbalanced, retry up to 3 times.
      After 3 failures, randomly assign sides.
   d. Recursively build left and right subtrees
   e. Create split node with hyperplane in `v[]`

### Phase 3: Root Discovery Hack
- Copy all root nodes to end of buffer
- Roots identified by `n_descendants == n_items`
- Enables loading without separate metadata

**Multithreading**: Each thread builds its own trees with a thread-specific
random seed. Three mutexes coordinate: node alloc, node access (shared), roots.

## Search Algorithm

Uses a **max-heap priority queue** across all trees simultaneously:

```
1. Push all tree roots onto priority queue with distance = +infinity
2. While |candidates| < search_k:
   a. Pop node with best (smallest) distance estimate
   b. If leaf (n_descendants == 1):
        Add item to candidate set
   c. If small intermediate (n_descendants <= K):
        Add all item IDs to candidate set
   d. If split node (n_descendants > K):
        margin = dot(query, split_vector)
        Push left child with  pq_distance(current_dist, margin, LEFT)
        Push right child with pq_distance(current_dist, margin, RIGHT)
3. Deduplicate candidates (items found via multiple trees)
4. Compute exact distances for all candidates
5. partial_sort to get top-k
```

**pq_distance** varies by metric but generally:
- Angular/Cosine: `min(d, margin)` for near side, `min(d, -margin)` for far side
- The margin estimates how far the query is from the split plane

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_trees` | user-specified | Number of trees. More = better recall, larger index |
| `search_k` | `n * n_trees` | Max nodes to visit during search. More = slower, better recall |

Typical: 10-100 trees, search_k = 10x-100x the number of neighbors requested.

## Distance Metrics

1. **Angular (Cosine)**: `sqrt(2 - 2*cos(u,v))`. Split plane = normal between
   two cluster centroids.
2. **Euclidean**: `sqrt(sum((x-y)^2))`. Split = perpendicular bisector with
   offset term `a`: `margin = a + dot(v, query)`.
3. **Manhattan**: `sum(|x-y|)`. Same Minkowski base as Euclidean.
4. **Dot Product**: Uses Microsoft Research transformation to convert to angular
   space by appending a dimension. Preprocessing computes global max norm.
5. **Hamming**: For binary vectors. Split on single bit position.

## Key Properties for SQLite Integration

- **Read-only after build**: Original Annoy is immutable after `build()`.
  Updates require full rebuild.
- **Fixed-size nodes**: Enables simple array-based storage. Each node is exactly
  `_s` bytes.
- **Node addressing**: Simple integer index into flat array. Node i is at
  offset `i * _s`.
- **Shared leaves**: All trees reference the same leaf nodes. Only internal
  nodes differ per tree.
- **No graph edges**: Unlike DiskANN, annoy has no neighbor lists to maintain.
  The tree structure IS the index.
- **Build is O(n log n) per tree**: Relatively fast, and embarrassingly parallel
  across trees.
- **Small intermediate optimization**: Avoids deep trees for small clusters.
