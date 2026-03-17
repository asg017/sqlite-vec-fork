# Annoy Index for sqlite-vec

## Overview

An annoy-based approximate nearest neighbor index for sqlite-vec's `vec0`
virtual table. Builds a forest of random binary search trees with split
hyperplanes, storing all tree structure in SQLite shadow tables. Queries
traverse all trees simultaneously via priority queue, collecting candidates,
then re-rank with exact distances.

Like IVF, the index requires an explicit build step after bulk insert. Unlike
DiskANN, there is no per-insert graph maintenance — the tree structure is
computed in batch.

## SQL API

### Table Creation

```sql
CREATE VIRTUAL TABLE vec_items USING vec0(
  id INTEGER PRIMARY KEY,
  embedding float[768] distance_metric=cosine
    INDEXED BY annoy(n_trees=50, search_k=200)
);

-- With quantization for smaller tree nodes
CREATE VIRTUAL TABLE vec_items USING vec0(
  id INTEGER PRIMARY KEY,
  embedding float[768] distance_metric=cosine
    INDEXED BY annoy(n_trees=100, search_k=400, quantizer=int8)
);

-- With partition key and metadata
CREATE VIRTUAL TABLE vec_items USING vec0(
  id INTEGER PRIMARY KEY,
  user_id INTEGER PARTITION KEY,
  category TEXT,
  +title TEXT,
  embedding float[768] distance_metric=cosine
    INDEXED BY annoy(n_trees=50, search_k=200)
);
```

### Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `n_trees` | 1-1000 | 50 | Number of trees in the forest. More = better recall, larger index |
| `search_k` | >= 1, or 0 | 0 | Max nodes to visit during search. 0 = auto (`k * n_trees * 10`) |
| `quantizer` | `none`, `int8`, `binary` | `none` | How split vectors are stored in tree nodes |

### Inserting Vectors

```sql
-- Vectors are stored immediately, but not yet indexed
INSERT INTO vec_items(id, embedding) VALUES (1, :vector);
```

Before the index is built, KNN queries fall back to brute-force scan.
After building, new inserts go to an "unindexed" buffer; queries search both
the tree index and the buffer.

### Building the Index

```sql
-- Build (or rebuild) the annoy forest from all vectors
INSERT INTO vec_items(id) VALUES ('build-index');

-- Rebuild only — clears existing trees first
INSERT INTO vec_items(id) VALUES ('rebuild-index');
```

This reads all vectors, builds n_trees random binary trees, and writes all
nodes to shadow tables. It's a blocking operation — run after bulk insert
or periodically as data changes.

### Runtime Parameter Tuning

```sql
-- Change search_k without rebuilding
INSERT INTO vec_items(id) VALUES ('search_k=500');
```

### KNN Queries

```sql
-- Standard vec0 KNN syntax
SELECT id, distance
FROM vec_items
WHERE embedding MATCH :query AND k = 10;

-- With metadata filter
SELECT id, distance
FROM vec_items
WHERE embedding MATCH :query AND k = 10
  AND category = 'science';

-- With partition key
SELECT id, distance
FROM vec_items
WHERE embedding MATCH :query AND k = 10
  AND user_id = 42;
```

### Deleting

```sql
DELETE FROM vec_items WHERE id = 123;
```

Deletes mark the item as removed. The tree structure becomes stale (deleted
items may still appear as candidates during traversal but are filtered out).
Periodic `rebuild-index` reclaims space.

## How It Works

### Architecture

```
Insert path:
  User vector → write to _annoy_vectors table
              → write to _annoy_buffer table (unindexed marker)
              → (after 'build-index') trees built from all vectors

Query path:
  Query vector → traverse all trees via priority queue
              → collect candidate item IDs (from descendant bitmaps)
              → look up full vectors from _annoy_vectors
              → compute exact distances
              → merge with buffer candidates (brute-force)
              → apply metadata/partition filters
              → return top k
```

### Shadow Tables

For table `vec_items` with vector column index `00`:

| Table | Schema | Purpose |
|-------|--------|---------|
| `vec_items_annoy_nodes00` | `node_id INTEGER PK, tree_id INTEGER, node_type INTEGER, data BLOB` | All tree nodes (split planes + descendants) |
| `vec_items_annoy_vectors00` | `rowid INTEGER PK, vector BLOB` | Full-precision vectors for distance computation |
| `vec_items_annoy_buffer00` | `rowid INTEGER PK, vector BLOB` | Vectors inserted after last build (unindexed) |
| `vec_items_annoy_meta00` | `key TEXT PK, value BLOB` | Index metadata (roots, config, item count, build state) |

Indices:
- `CREATE INDEX idx_annoy_nodes_tree ON _annoy_nodes00(tree_id, node_id)` —
  enables efficient per-tree traversal

### Node Storage Format

Each row in `_annoy_nodes00` stores one tree node:

**node_type = 0 (Split Node):**
```
data BLOB layout:
  [4 bytes: left_node_id (int32)]
  [4 bytes: right_node_id (int32)]
  [N bytes: split vector (float32 * dimensions, or quantized)]
```

For Euclidean/Manhattan distance, an additional 4-byte offset term is prepended
to the split vector.

**node_type = 1 (Descendants / Leaf Bucket):**
```
data BLOB layout:
  [N bytes: packed int64 array of item rowids]
```

For large datasets, consider using a roaring bitmap encoding instead of packed
arrays when descendant count exceeds a threshold.

**node_type = 2 (Root Pointer — stored in _annoy_meta instead):**
Not a real node type. Root node IDs are stored in the metadata table.

### Metadata Table

The `_annoy_meta00` table stores:

| Key | Value | Description |
|-----|-------|-------------|
| `roots` | BLOB (packed int32 array) | Node IDs of all tree roots |
| `n_trees` | INTEGER | Number of trees built |
| `n_items` | INTEGER | Number of items at last build |
| `search_k` | INTEGER | Current search_k setting |
| `built` | INTEGER | 1 if index has been built |
| `quantizer` | TEXT | Quantizer type used |

### Quantization for Split Vectors

Split vectors in tree nodes can be quantized to reduce storage:

- **none**: float32 split vectors (full precision)
- **int8**: 4x smaller nodes, marginal recall impact (split planes are
  approximate anyway)
- **binary**: 32x smaller nodes, more recall impact but still usable

Query vectors are always full precision. Only the stored split hyperplanes
are quantized.

Note: The full-precision item vectors in `_annoy_vectors` are never quantized —
exact distances are always computed from originals.

## Build Algorithm

### Step 1: Read All Vectors

Load all vectors from `_annoy_vectors00` (or from chunk storage) into a
working buffer. For datasets larger than available memory, build trees in
a streaming/sampling fashion (future optimization).

### Step 2: Build Trees

For each tree (1..n_trees), independently:

```
function build_tree(item_ids):
    if len(item_ids) <= MAX_DESCENDANTS (e.g., 128):
        node_id = allocate_node()
        write descendants node (node_type=1, data=packed item_ids)
        return node_id

    // Find split hyperplane via two-means
    p, q = random_sample(item_ids, 2)
    for 200 iterations:
        k = random_sample(item_ids, 1)
        assign k to closer of p, q (weighted)
    split_vector = normalize(p - q)

    // Partition items
    left_ids  = [id for id in item_ids if dot(vector[id], split) < 0]
    right_ids = [id for id in item_ids if dot(vector[id], split) >= 0]

    // Handle imbalanced splits (>95% one side): retry up to 3x, then randomize
    if imbalanced:
        retry or random_shuffle

    left_child  = build_tree(left_ids)
    right_child = build_tree(right_ids)

    node_id = allocate_node()
    write split node (node_type=0, data=[left, right, split_vector])
    return node_id
```

### Step 3: Write Metadata

Store root node IDs, tree count, item count, and set `built = 1`.

### Step 4: Clear Buffer

Move all buffer entries to the indexed state (or just clear the buffer table
since vectors are already in `_annoy_vectors`).

### Build Performance Estimate

Each tree is O(n log n). Total build: O(n_trees * n * log n).
Trees are independent and can be built sequentially (or in parallel if
we add threading later — but SQLite write transactions are serial anyway).

For 1M vectors at 768 dimensions with 50 trees: expect ~30-120 seconds
depending on hardware.

## Search Algorithm

```
function annoy_search(query, k, search_k):
    if search_k == 0:
        search_k = k * n_trees * 10

    // Load roots from metadata
    roots = load_roots()

    // Priority queue: (distance_estimate, node_id)
    // Max-heap by distance — pop gives closest unexplored node
    pq = new PriorityQueue()
    for root in roots:
        pq.push(+infinity, root)

    candidates = new Set()

    while len(candidates) < search_k and pq not empty:
        (dist, node_id) = pq.pop()
        node = load_node(node_id)   // single SELECT from _annoy_nodes

        if node.type == DESCENDANTS:
            candidates.add_all(node.item_ids)
        elif node.type == SPLIT:
            margin = dot(query, node.split_vector)
            // Closer side gets better priority
            pq.push(pq_distance(dist, margin, LEFT),  node.left_child)
            pq.push(pq_distance(dist, margin, RIGHT), node.right_child)

    // Also scan buffer for unindexed items
    buffer_candidates = scan_buffer(query)
    candidates.merge(buffer_candidates)

    // Deduplicate
    candidates = unique(candidates)

    // Compute exact distances
    results = []
    for rowid in candidates:
        vec = load_vector(rowid)     // SELECT from _annoy_vectors
        dist = exact_distance(query, vec)
        results.append((rowid, dist))

    // Return top k
    partial_sort(results, k)
    return results[:k]
```

### Optimization: Batch Node Loading

Instead of one SELECT per node, we can batch-load nodes:
- Collect a batch of node_ids from the priority queue
- `SELECT * FROM _annoy_nodes WHERE node_id IN (...)`
- Process all in one round-trip

Similarly for vector lookups during re-ranking:
- `SELECT * FROM _annoy_vectors WHERE rowid IN (...)`

### Optimization: Node Caching

Frequently accessed nodes (roots, upper levels of trees) could be cached
in-memory on the `vec0_vtab` struct. Upper tree levels are accessed on every
query and are a small fraction of total nodes.

### Search Performance Estimate

For 1M vectors, 50 trees, search_k=500:
- ~500 node lookups (mostly B-tree point reads)
- ~200-400 unique candidates after dedup
- ~200-400 vector lookups for exact distance
- Expected: 5-30ms per query depending on caching and I/O

## Integration with Metadata / Partition / Auxiliary Columns

### Partition Keys

When partition keys are present, the annoy index should be **per-partition**.
Each partition has its own set of trees.

Options (in order of preference):
1. **Composite tree_id**: Include partition key in the node table schema
   and tree lookup. E.g., add `partition_key` column to `_annoy_nodes`
   and `_annoy_meta`.
2. **Separate meta rows per partition**: `_annoy_meta` keys become
   `roots:{partition_value}`, etc.

Queries with partition constraints only search that partition's trees.
Queries without partition constraints search all partitions (slower).

### Metadata Columns (Filterable)

Metadata filters are applied **post-retrieval**: the annoy search returns
candidate rowids, then metadata filters are checked before returning results.

To maintain target recall with filtering, oversample: collect more candidates
than k, apply filters, then take top k from remaining.

The existing vec0 metadata filtering infrastructure handles this — the annoy
search just needs to return a larger candidate set.

### Auxiliary Columns

No interaction with the annoy index. Auxiliary columns are stored separately
and retrieved after KNN results are determined.

## Implementation Plan

### File Structure

```
sqlite-vec-annoy.c          All annoy logic: parser, shadow tables, build,
                             search, insert/delete hooks, node encode/decode
sqlite-vec.c                 ~50 lines of additions: struct fields, #includes,
                             dispatch hooks in parse/create/insert/delete/filter
```

`sqlite-vec-annoy.c` is `#include`d into `sqlite-vec.c`, following the
DiskANN and IVF pattern.

### Step-by-Step Implementation

#### Step 1: Data Structures and Constants

Add to `sqlite-vec.c`:

```c
// Annoy configuration
struct Vec0AnnoyConfig {
    int enabled;
    int n_trees;            // default 50
    int search_k;           // default 0 (auto)
    enum Vec0AnnoyQuantizerType quantizer_type;  // NONE, INT8, BINARY
};

// Node types
#define VEC0_ANNOY_NODE_SPLIT 0
#define VEC0_ANNOY_NODE_DESCENDANTS 1

// Limits
#define VEC0_ANNOY_MAX_DESCENDANTS 128  // leaf bucket size
#define VEC0_ANNOY_MAX_TREES 1000
```

Add fields to `vec0_vtab`:
```c
struct Vec0AnnoyConfig annoy_configs[VEC0_MAX_VECTOR_COLUMNS];
// Shadow table names
char *shadowAnnoyNodesNames[VEC0_MAX_VECTOR_COLUMNS];
char *shadowAnnoyVectorsNames[VEC0_MAX_VECTOR_COLUMNS];
char *shadowAnnoyBufferNames[VEC0_MAX_VECTOR_COLUMNS];
char *shadowAnnoyMetaNames[VEC0_MAX_VECTOR_COLUMNS];
// Cached prepared statements
sqlite3_stmt *stmtAnnoyNodeRead[VEC0_MAX_VECTOR_COLUMNS];
sqlite3_stmt *stmtAnnoyNodeWrite[VEC0_MAX_VECTOR_COLUMNS];
sqlite3_stmt *stmtAnnoyVectorRead[VEC0_MAX_VECTOR_COLUMNS];
sqlite3_stmt *stmtAnnoyVectorWrite[VEC0_MAX_VECTOR_COLUMNS];
sqlite3_stmt *stmtAnnoyBufferRead[VEC0_MAX_VECTOR_COLUMNS];
sqlite3_stmt *stmtAnnoyBufferWrite[VEC0_MAX_VECTOR_COLUMNS];
sqlite3_stmt *stmtAnnoyMetaRead[VEC0_MAX_VECTOR_COLUMNS];
sqlite3_stmt *stmtAnnoyMetaWrite[VEC0_MAX_VECTOR_COLUMNS];
```

#### Step 2: Parse `INDEXED BY annoy(...)` Syntax

In `vec0_parse_vector_column()`, after existing DiskANN/IVF parsing:

```c
if (scanner_match("annoy")) {
    parse_annoy_options(scanner, &config);
    // Validate: n_trees in range, search_k >= 0, quantizer valid
    column_def->annoy = config;
}
```

#### Step 3: Create Shadow Tables

In `vec0_init()` (xCreate branch), for each annoy-enabled column:

```sql
CREATE TABLE IF NOT EXISTS "{schema}"."{table}_annoy_nodes{NN}" (
    node_id INTEGER PRIMARY KEY,
    tree_id INTEGER NOT NULL,
    node_type INTEGER NOT NULL,
    data BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS "{schema}"."idx_{table}_annoy_nodes_tree{NN}"
    ON "{table}_annoy_nodes{NN}"(tree_id);

CREATE TABLE IF NOT EXISTS "{schema}"."{table}_annoy_vectors{NN}" (
    rowid INTEGER PRIMARY KEY,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS "{schema}"."{table}_annoy_buffer{NN}" (
    rowid INTEGER PRIMARY KEY,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS "{schema}"."{table}_annoy_meta{NN}" (
    key TEXT PRIMARY KEY,
    value BLOB
);
```

Register shadow table names in `vec0ShadowName()`.

#### Step 4: Node Blob Encode/Decode

```c
// Encode split node
int annoy_encode_split_node(
    i32 left_id, i32 right_id,
    const float *split_vector, int dimensions,
    enum quantizer_type qt,
    unsigned char *out, int *out_len
);

// Encode descendants node
int annoy_encode_descendants_node(
    const i64 *rowids, int n_rowids,
    unsigned char *out, int *out_len
);

// Decode node (returns type, populates union)
struct AnnoyNode {
    int type;
    union {
        struct { i32 left, right; float *split_vector; } split;
        struct { i64 *rowids; int n_rowids; } descendants;
    };
};
int annoy_decode_node(const unsigned char *data, int len, int dimensions, AnnoyNode *out);
```

#### Step 5: Two-Means Split Function

```c
// Find split hyperplane using two-means heuristic
int annoy_two_means_split(
    const float *vectors,    // all vectors (flat array)
    const i64 *item_ids,     // items to split
    int n_items,
    int dimensions,
    float *split_vector_out, // output: normalized split hyperplane
    i64 *left_ids, int *n_left,
    i64 *right_ids, int *n_right
);
```

#### Step 6: Recursive Tree Build

```c
// Build one tree recursively, writing nodes to shadow table
i32 annoy_build_tree(
    vec0_vtab *p,
    int col_idx,
    const float *vectors,
    i64 *item_ids,
    int n_items,
    int tree_id,
    i32 *next_node_id
);
```

#### Step 7: Build Command Handler

Wire `'build-index'` and `'rebuild-index'` commands into the existing
special-insert dispatch (same pattern as IVF's `'compute-centroids'`):

```c
// In vec0Update_Insert, check for special command strings:
if (strcmp(cmd, "build-index") == 0) {
    return annoy_build_all(p);
}
if (strcmp(cmd, "rebuild-index") == 0) {
    annoy_clear_trees(p);
    return annoy_build_all(p);
}
```

`annoy_build_all()`:
1. Read all vectors from `_annoy_vectors` into memory
2. For each tree 0..n_trees-1: call `annoy_build_tree()`
3. Write roots to `_annoy_meta`
4. Clear `_annoy_buffer`
5. Set `built = 1` in meta

#### Step 8: Insert Hook

In `vec0Update_Insert()`, after writing chunk data:

```c
if (p->annoy_configs[col].enabled) {
    // Write full-precision vector to _annoy_vectors
    annoy_store_vector(p, col, rowid, vector_data);
    // Also add to buffer (unindexed since last build)
    annoy_buffer_add(p, col, rowid, vector_data);
}
```

#### Step 9: Delete Hook

```c
if (p->annoy_configs[col].enabled) {
    // Remove from _annoy_vectors
    annoy_remove_vector(p, col, rowid);
    // Remove from buffer if present
    annoy_buffer_remove(p, col, rowid);
    // Note: stale tree references are handled at query time
    // (candidate lookup fails → skip)
}
```

#### Step 10: Search Function

```c
int annoy_search(
    vec0_vtab *p,
    int col_idx,
    const float *query,
    int k,
    int search_k,
    // Output
    i64 *out_rowids,
    float *out_distances,
    int *out_n_results
);
```

Priority queue implementation (min-heap by negative distance for max-heap
behavior, or use a proper max-heap).

#### Step 11: Wire Search into KNN Query Path

In `vec0Filter()` KNN plan handler:

```c
if (p->annoy_configs[col].enabled && annoy_is_built(p, col)) {
    rc = vec0Filter_knn_annoy(cursor, col, query_vec, k, ...);
} else {
    // Fall back to brute-force
    rc = vec0Filter_knn_flat(cursor, col, query_vec, k, ...);
}
```

#### Step 12: Runtime Parameter Commands

```c
if (starts_with(cmd, "search_k=")) {
    int new_sk = atoi(cmd + 9);
    p->annoy_configs[col].search_k = new_sk;
    annoy_meta_set_int(p, col, "search_k", new_sk);
}
```

### Testing Strategy

1. **Unit tests**: Node encode/decode, two-means split, single tree build
2. **Integration tests**: Full INSERT → build-index → KNN query cycle
3. **Recall benchmarks**: Compare against brute-force at various n_trees/search_k
4. **Stress tests**: Large datasets (100k, 1M), concurrent queries
5. **Edge cases**: Empty index, single item, duplicate vectors, delete-heavy
6. **Partition tests**: Per-partition tree building and querying
7. **Parameter sweep**: n_trees vs recall curves, search_k vs latency

### Benchmark Targets

For 1M vectors, 768 dimensions, cosine distance:

| Config | Query Time | Recall@10 | Index Size |
|--------|-----------|-----------|------------|
| annoy(n_trees=10) | ~5ms | ~0.80 | ~1.5x vectors |
| annoy(n_trees=50) | ~15ms | ~0.93 | ~3x vectors |
| annoy(n_trees=100) | ~25ms | ~0.97 | ~5x vectors |
| annoy(n_trees=50, q=int8) | ~12ms | ~0.91 | ~1.5x vectors |

These are rough estimates based on annoy's known characteristics. Actual
performance depends heavily on SQLite B-tree I/O patterns vs. mmap.

## Design Decisions and Rationale

### Why batch build instead of per-insert tree maintenance?

1. **Algorithmic simplicity**: Annoy's trees are built top-down from a
   complete dataset. Incremental insertion into a random projection tree
   would require re-splitting, which changes the tree structure unpredictably.
2. **Matches user expectations**: Annoy is traditionally a build-once index.
   The buffer provides a graceful degradation for new inserts.
3. **Precedent**: IVF uses the same pattern (`compute-centroids` command).

### Why a buffer table for unindexed items?

After building, new inserts can't be placed in the tree without rebuilding.
The buffer provides:
- Immediate queryability (brute-force over buffer + tree search)
- Clear signal to users when a rebuild is beneficial (buffer growing)
- No data loss between builds

### Why store full vectors separately from chunks?

The annoy search needs to compute exact distances for candidates. Reading
individual vectors from packed 64-item chunks requires unpacking. A simple
KV table (`rowid → vector`) gives O(1) random access without chunk overhead.

This matches the DiskANN and IVF patterns which both have `_vectors{NN}` tables.

### Why not store nodes like annoy's flat file?

Annoy stores all nodes in a flat array with fixed-size entries. We could
store this as a single BLOB or a set of page-sized BLOBs. However:

1. **Variable-size nodes are more space-efficient**: Descendant nodes with
   10 items don't need space for a 768-dim split vector.
2. **SQLite B-tree is efficient for point lookups**: Each node access is
   a single B-tree lookup, which is O(log n) but with excellent caching.
3. **Transactional safety**: Individual rows can be updated/deleted without
   rewriting the entire structure.
4. **Matches arroy's proven approach**: Arroy stores nodes as individual
   KV pairs in LMDB and performs well.

### Why tree_id column instead of encoding in node_id?

Having `tree_id` as a separate column enables:
- Easy deletion of a single tree (`DELETE WHERE tree_id = X`)
- Efficient per-tree iteration (via index)
- Clear semantics in the schema

The cost (one extra integer column) is negligible.

## Future Optimizations

1. **Node caching**: Cache upper tree levels in memory (small, accessed every query)
2. **Batch node reads**: Prefetch node batches instead of one-at-a-time
3. **Parallel tree building**: Build trees in separate transactions (requires care)
4. **Incremental rebuild**: Only rebuild trees affected by many deletes
5. **Streaming build**: Build trees from sampled subsets for very large datasets
6. **SIMD distance**: Use NEON/AVX for split margin computation and exact distances
   (may already be available from existing sqlite-vec SIMD infrastructure)
7. **Adaptive search_k**: Automatically increase search_k when buffer is large
