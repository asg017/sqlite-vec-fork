# Annoy Query Profiling Analysis

## Setup

- Dataset: COHERE 768-dim cosine
- Tool: `perf record -g --call-graph dwarf` on Linux
- Built with `-g3 -O2` (debug symbols + optimization)
- Two workloads: 10k vectors (200 queries) and 50k vectors (50 queries)
- n_trees=25, search_k=auto (k * n_trees * 10 = 2500)

## High-Level Breakdown

| Category | 10k (%) | 50k (%) |
|----------|---------|---------|
| Kernel (disk I/O, page cache) | 62% | 64% |
| Userspace (sqlite-vec + SQLite) | 29% | 28% |
| BPF/tracing overhead | 2% | 2% |

**~64% of query time is spent in kernel disk I/O.** This is the dominant cost.

## Top Userspace Functions (10k)

```
10.32%  distance_cosine_float         -- exact distance computation for re-ranking
 2.90%  __libc_pread                   -- system call for reading pages
 2.43%  vec0Filter_knn                 -- query dispatch/orchestration
 2.14%  sqlite3BtreeTableMoveto        -- B-tree point lookup per node
 1.51%  sqlite3VdbeExec                -- SQLite VM execution
 1.22%  pcache1Fetch                   -- page cache lookup
 0.87%  sqlite3VdbeHalt                -- statement reset between queries
 0.81%  getPageNormal                  -- page loading
 0.58%  annoy_candidate_cmp            -- qsort comparator for results
 0.46%  annoy_cmp_i64                  -- qsort comparator for dedup
 0.40%  annoy_vector_read              -- vector lookup from _annoy_vectors
```

## Bottleneck Analysis

### 1. Disk I/O dominates (64%)
- `_copy_to_iter` (17%) + `filemap_get_read_batch` (7%) + `filemap_read` (2%)
- These are kernel page cache operations for reading SQLite pages
- Each `annoy_node_read()` call triggers a B-tree lookup → page read
- Each `annoy_vector_read()` for re-ranking also triggers a page read
- With search_k=2500, we do ~2500 node reads + ~500 vector reads per query

### 2. Cosine distance computation (10%)
- `distance_cosine_float` is the re-ranking step: computing exact cosine
  distance between query and each candidate vector
- 768-dim dot product + magnitudes = ~2300 FLOPs per candidate
- ~200-500 candidates per query = reasonable
- Could be improved with SIMD (AVX/NEON) but not the main bottleneck

### 3. B-tree lookups (5%)
- `sqlite3BtreeTableMoveto` (2-3%) + `pcache1Fetch` (1%) + `getPageNormal` (1%)
- Each node/vector lookup does a B-tree traversal to find the row
- This is inherent to storing nodes as individual SQLite rows

### 4. SQLite VM overhead (3%)
- `sqlite3VdbeExec` (1.5%) + `sqlite3VdbeHalt` (1%)
- The prepared statement machinery for each node read
- Minimal — this is already cached via lazy statement preparation

## Why 50k is Much Slower Than 10k

At 10k: 8.8ms/query. At 50k: 12ms/query (for same n_trees=25).

The scaling is sub-linear but the constant factor is high because:
1. More nodes per tree (deeper trees) → more node reads per search
2. Larger database → less fits in page cache → more cold reads
3. search_k=2500 constant means same exploration depth, but nodes are spread
   across more pages

## Optimization Priorities

### High Impact (address the 64% I/O bottleneck)

1. **Batch node reads**: Instead of one `SELECT` per node during PQ traversal,
   collect a batch of node_ids and read them in one `WHERE node_id IN (...)`
   query. This allows SQLite to plan sequential page reads.

2. **Cache upper tree nodes in memory**: The root and first few levels of each
   tree are accessed on every query. Cache them on the `vec0_vtab` struct
   to avoid repeated B-tree lookups.

3. **Batch vector reads for re-ranking**: After collecting all candidates,
   read their vectors in one batch query instead of one-by-one.

4. **PRAGMA mmap_size**: Enable memory-mapped I/O for the database to avoid
   the pread syscall overhead entirely.

### Medium Impact (address the 10% distance computation)

5. **SIMD cosine distance**: Use AVX2 or NEON intrinsics for the 768-dim
   dot product. Could give 4-8x speedup on the distance computation.
   (Already available in sqlite-vec for other code paths)

### Low Impact (address the 5% B-tree overhead)

6. **Pack nodes more efficiently**: Store nodes in a flat BLOB per tree
   instead of individual rows. One large blob read vs many small B-tree lookups.
   This is a bigger architectural change but would eliminate B-tree overhead.
