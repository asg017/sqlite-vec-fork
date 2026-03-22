# Annoy Index TODO / Optimization Roadmap

## Current Performance (post batch-read optimization)

```
         Config   10k qry  10k recall  50k qry  50k recall  100k qry  100k recall
       annoy-t25   11.8ms      0.872   15.9ms      0.778    16.4ms       0.722
       annoy-t50   14.8ms      0.958   21.2ms      0.906    48.5ms       0.894
      annoy-t100   23.4ms      0.988       -          -     746.3ms       0.948
    brute-float    17.9ms      1.000   89.8ms      1.000   174.9ms       1.000
     brute-int8    21.6ms      0.998   59.1ms      0.998   109.4ms       1.000
      brute-bit    14.5ms      0.852   21.3ms      0.846    28.9ms       0.884
```

## Profiling summary (from perf record on Linux)

CPU time breakdown for KNN queries (10k, n_trees=25):
- 64% kernel disk I/O (page cache reads via pread syscall)
- 10% distance_cosine_float (exact distance re-ranking, scalar — no SIMD)
-  5% SQLite B-tree machinery (BtreeTableMoveto, pcache1Fetch, getPageNormal)
-  3% SQLite VM overhead (VdbeExec, VdbeHalt)
-  2% annoy-specific (candidate sort, dedup, vector_read dispatch)

The batched json_each reads eliminated the per-node prepare/finalize overhead
and reduced I/O round-trips by 64x (batch of 64 nodes per query). This gave
the 18x speedup at 50k. But I/O is still dominant — each batch still triggers
B-tree traversals and page reads.

---

## Priority 1: Cache upper tree levels in memory

**Problem:** Every KNN query reads the same root nodes and upper split nodes
from SQLite. At n_trees=50, that's 50 root reads + ~100-200 upper-level node
reads per query, all hitting the B-tree every time.

**Approach:** On first query (or on build), load root nodes + first N levels
of each tree into an in-memory cache on the `vec0_vtab` struct.

```c
// In vec0_vtab:
struct AnnoyNodeCache {
    int n_cached;
    i32 *node_ids;
    int *node_types;
    u8 **data;
    int *data_sizes;
} annoyNodeCache[VEC0_MAX_VECTOR_COLUMNS];
```

**Expected impact:** At n_trees=25, caching 3 levels means caching
~25 + 50 + 100 = 175 nodes. Each is ~3KB (768 floats). Total ~525KB per
column — fits easily in memory. Eliminates ~175 B-tree lookups per query.

**Notes:**
- Cache should be invalidated on rebuild-index
- Cache only split nodes (not descendants, which are leaf-level)
- Could also cache the roots blob from _info to avoid that read too
- The annoy_info_get_roots() call on every query is wasteful — cache it

**Where to implement:**
- `annoy_search()` currently calls `annoy_info_get_roots()` per query — cache this
- The PQ traversal batch read can check the cache before hitting SQLite
- Add a `annoy_cache_init()` called from `annoy_build_all()` or lazily on first query
- Add cleanup in `vec0_free_resources()`

---

## Priority 2: PRAGMA mmap_size for memory-mapped I/O

**Problem:** 64% of query time is kernel I/O. The current path is:
```
annoy_node_read → sqlite3_step → B-tree lookup → unixRead → pread syscall
  → kernel: filemap_read → _copy_to_iter → copy page to userspace
```

With mmap, the path becomes:
```
annoy_node_read → sqlite3_step → B-tree lookup → direct memory access (no syscall)
```

**Approach:** Set `PRAGMA mmap_size = <file_size>` when opening the database.
This tells SQLite to memory-map the entire file, avoiding pread syscalls.

**Notes:**
- This is a user-facing configuration, not something we set automatically
- Could document it as a recommended PRAGMA for annoy-indexed tables
- Alternatively, set it in vec0_init when annoy is enabled:
  ```c
  if (has_annoy_columns) {
      sqlite3_exec(db, "PRAGMA mmap_size=1073741824", NULL, NULL, NULL);  // 1GB
  }
  ```
- Risk: mmap can cause issues with WAL mode on some platforms
- The IVF TODO.md also identified this as a potential optimization
- At 100k, the DB is ~1.6GB — may not fully fit in mmap on low-memory systems
- Could use a more conservative default and let users tune it up

**Expected impact:** Could eliminate most of the 64% I/O overhead for
warm queries. First query still cold, but subsequent queries use page cache
via mmap instead of syscalls. Estimated 2-4x speedup for repeated queries.

---

## Priority 3: SIMD cosine distance

**Problem:** `distance_cosine_float` is 10% of query time and uses a scalar
loop:
```c
for (size_t i = 0; i < qty; i++) {
    dot += *pVect1 * *pVect2;
    aMag += *pVect1 * *pVect1;
    bMag += *pVect2 * *pVect2;
    pVect1++; pVect2++;
}
```

No SIMD path exists for cosine in sqlite-vec. The AVX path only covers L2
(`distance_l2_sqr_float`). NEON covers L2, L1, and cosine — but only on ARM.

**Approach:** Add `cosine_float_avx()` using 256-bit AVX intrinsics:
```c
// Process 8 floats at a time
__m256 sum_dot = _mm256_setzero_ps();
__m256 sum_a   = _mm256_setzero_ps();
__m256 sum_b   = _mm256_setzero_ps();
for (size_t i = 0; i < qty; i += 8) {
    __m256 a = _mm256_loadu_ps(pVect1 + i);
    __m256 b = _mm256_loadu_ps(pVect2 + i);
    sum_dot = _mm256_fmadd_ps(a, b, sum_dot);  // dot += a * b
    sum_a   = _mm256_fmadd_ps(a, a, sum_a);    // aMag += a * a
    sum_b   = _mm256_fmadd_ps(b, b, sum_b);    // bMag += b * b
}
// horizontal sum each accumulator...
```

At 768 dims, this processes 96 iterations instead of 768. With FMA, that's
~3 FLOPs per element per iteration → ~2300 FLOPs in 96 iterations vs 768.

**Notes:**
- This machine has AVX512 — could use 512-bit for 16 floats/iteration
- _mm256_fmadd_ps requires FMA3 (available on this CPU)
- Need to add `-mfma` to CFLAGS alongside `-mavx`
- The Makefile only enables AVX on macOS x86_64 — should also enable on Linux
- This benefits ALL cosine queries, not just annoy
- The NEON cosine implementation on the ivf-yolo branch (`cosine_float_neon`)
  can be used as a reference for the AVX version

**Expected impact:** 4-8x speedup on the cosine computation, which is 10% of
total time → ~8-9% overall speedup. Not huge, but it's free performance and
benefits all vec0 queries.

**Where to implement:**
- Add `cosine_float_avx()` in the SIMD section (~line 120-200)
- Wire into `distance_cosine_float()` with `#ifdef SQLITE_VEC_ENABLE_AVX`
- Update Makefile to enable AVX on Linux: `ifeq ($(shell uname -s),Linux)`

---

## Priority 4: Adaptive batch sizes / search_k tuning

**Problem:** annoy-t100 at 100k takes 746ms because search_k = 10 * 100 * 10
= 10000. With batch size 64, that's ~156 batch SQL queries during PQ
traversal, each doing a json_each + B-tree scan. The overhead compounds.

**Approaches:**

### 4a. Larger batch sizes
Current: `ANNOY_NODE_BATCH_SIZE = 64`. Could increase to 256 or 512.
Trade-off: larger batches read more nodes than needed (some may not be
explored by the PQ). But the per-batch overhead is fixed (prepare + execute),
so fewer batches = less overhead.

### 4b. Smarter search_k auto formula
Current: `search_k = k * n_trees * 10`. This is very aggressive for high
n_trees. Original annoy uses `search_k = k * n_trees` (no 10x multiplier).
We could:
- Use `search_k = k * n_trees` (match annoy default)
- Cap search_k at e.g. 5000 regardless of n_trees
- Let users tune it at runtime (already supported via `search_k=N` command)

### 4c. Early termination
Stop PQ traversal when the k-th best candidate distance is better than
the PQ's best remaining distance estimate. This is what annoy does implicitly
via the max-heap — once the heap's best entry is worse than our k-th result,
no more candidates can improve the result.

Currently we don't track the k-th best distance during traversal. Adding this
would require maintaining a top-k heap alongside the PQ, which is more
complexity but would significantly reduce wasted work at high search_k.

**Notes:**
- The 10x multiplier was added empirically to compensate for the originally
  broken PQ priority. Now that it's fixed, the default multiplier could be
  reduced to 3-5x.
- At 100k, reducing from 10x to 3x would cut search_k from 10000 to 3000,
  which should bring annoy-t100 from 746ms to ~220ms (linear in search_k).
- Recall would decrease somewhat — need to benchmark the trade-off.

---

## Priority 5: Reduce DB size

**Problem:** DB size is 5-8x larger than baselines:
- 100k brute-float: 296 MB
- 100k annoy-t50: 1604 MB
- 100k annoy-t100: 2425 MB

The overhead comes from:
1. `_annoy_vectors` table: full 768-dim float32 vectors (one row per vector)
   = ~300MB at 100k. This duplicates the data in brute-force's chunk storage.
2. `_annoy_nodes` table: split nodes store 768-dim float32 hyperplanes each.
   At 100k with t50 and MAX_DESCENDANTS=48, there are ~2000 split nodes per
   tree × 50 trees = 100k nodes × 3KB each = ~300MB of split vectors.
3. `_annoy_buffer` table: empty after build, negligible.
4. Core shadow tables (_rowids, _chunks, _info): ~50MB overhead.

**Approaches:**

### 5a. Quantize split vectors
Split hyperplanes don't need full float32 precision. The margin computation
`dot(query, split_vector)` only needs to determine the sign (which side of
the hyperplane the query is on). Using int8 quantization for split vectors
would reduce node storage by 4x (~75MB instead of 300MB).

This was planned in the original ANNOY_PLAN.md as the `quantizer` parameter
but never implemented. The annoy config already parses it but doesn't use it.

### 5b. Share vectors with chunk storage
Currently annoy columns skip `_vector_chunks` and use `_annoy_vectors` instead.
An alternative: keep chunk storage and read vectors from chunks during
re-ranking. This avoids duplicating all vector data.

Trade-off: chunk reads require unpacking (offset into a 64-vector blob) vs
direct KV lookup from `_annoy_vectors`. The KV lookup is simpler and works
better with batch reads.

### 5c. Pack nodes more densely
Instead of one SQLite row per node, store all nodes for a tree in a single
BLOB. This eliminates B-tree overhead per node and reduces page waste.
Trade-off: can't do efficient single-node lookups (need to scan the blob).
But with caching, the blob would be loaded once and indexed in memory.

---

## Priority 6: Incremental rebuild

**Problem:** After deletes, the tree structure becomes stale — deleted items
appear as candidates during traversal and are filtered out during re-ranking
(vector not found → skipped). This wastes traversal work.

After inserts, new items go to the buffer and are brute-force scanned. As the
buffer grows, query time degrades linearly.

**Approach:** Periodic or threshold-triggered rebuild. Options:
- Automatic rebuild when buffer exceeds N% of total vectors
- Background rebuild in a separate transaction
- Partial rebuild: only rebuild trees where >M% of descendants are stale

**Notes:**
- Arroy (Rust annoy-in-LMDB) implements incremental updates by traversing
  existing trees and modifying descendant bitmaps. This is more complex but
  avoids full rebuilds.
- For v1, recommend documenting the `rebuild-index` command and suggesting
  users call it periodically.

---

## Priority 7: Partition-aware trees

**Problem:** The ANNOY_PLAN.md specifies per-partition tree building, but this
isn't implemented. Currently all vectors from all partitions share the same
trees.

**Approach:** When partition keys are present, store separate trees per
partition value. The `_annoy_nodes` table already has `tree_id` — extend the
metadata to track which tree_ids belong to which partition.

**Notes:**
- This is important for multi-tenant use cases
- Queries with partition constraints would only search that partition's trees
- Queries without partition constraints would need to search all partitions
- Build time increases linearly with number of partitions
- Could defer this to a later version

---

## Priority 8: Split vector quantization (annoy quantizer parameter)

**Problem:** Each split node stores a full 768-dim float32 hyperplane = 3072
bytes. At 100k vectors with n_trees=50, that's ~100k split nodes × 3KB =
~300MB just for hyperplanes. This is the single largest contributor to the
5-8x DB size overhead vs baselines.

**What was planned:** The ANNOY_PLAN.md specified a `quantizer` parameter:
```sql
INDEXED BY annoy(n_trees=50, quantizer=int8)
```

The `Vec0AnnoyConfig` struct does NOT currently have a quantizer field. The
parser does NOT accept a quantizer option. This needs to be added end-to-end.

**Approach:** Quantize the split hyperplane vector before storing in the node
blob. During search, dequantize (or compute the margin directly on quantized
data).

### int8 quantization (4x compression)
- Store: `split_vector_int8[i] = clamp(split_vector[i] * 127, -127, 127)`
- Margin computation: `dot(query_float, split_int8) / 127.0` — the sign is
  preserved, which is all that matters for tree traversal
- Node size: 768 bytes instead of 3072 bytes
- Expected recall impact: minimal — the hyperplane direction is well-preserved
  by int8, and the margin sign is robust to small quantization errors
- Already proven in DiskANN's neighbor quantization with minimal recall loss

### binary quantization (32x compression)
- Store: `split_vector_bit[i] = (split_vector[i] >= 0) ? 1 : 0`
- Margin computation: for each bit, accumulate `query[i]` if bit is 1, else
  subtract `query[i]`. Equivalent to `dot(query, sign(split_vector))`.
- Node size: 96 bytes instead of 3072 bytes (768 bits = 96 bytes)
- Expected recall impact: moderate — the hyperplane direction loses magnitude
  information, but sign is a reasonable proxy for high-dim data
- Need to benchmark recall impact before recommending

### Implementation steps:
1. Add `enum Vec0AnnoyQuantizerType` (NONE=0, INT8=1, BINARY=2) to config
2. Add `quantizer` field to `Vec0AnnoyConfig`
3. Parse `quantizer=none|int8|binary` in `vec0_parse_annoy_options()`
4. Update `annoy_encode_split_node()` to accept quantizer type and quantize
   the split vector before storing
5. Update `annoy_decode_split_node()` to dequantize on read (or compute
   margin directly on quantized data)
6. For int8: can reuse `diskann_quantize_vector()` (already exists for
   DiskANN neighbor quantization)
7. For binary: can reuse `vec_quantize_binary()` logic
8. Update internal header and unit tests

**Expected impact on DB size:**
- 100k t50 with int8: ~1604MB → ~1380MB (save ~225MB from split vectors)
- 100k t50 with binary: ~1604MB → ~1330MB (save ~275MB)
- Not as dramatic as hoped because `_annoy_vectors` (300MB) is the other big
  contributor and is NOT quantized (full precision needed for re-ranking)

**Expected impact on recall:**
- int8: <1% recall drop (DiskANN experience shows int8 is very good)
- binary: 2-5% recall drop — needs benchmarking
- Note: this only affects tree traversal accuracy, not re-ranking accuracy
  (re-ranking always uses full-precision vectors)

---

## Priority 9: Euclidean/Manhattan split plane offset

**Problem:** For L2 and L1 distance metrics, the split hyperplane needs an
offset term. In annoy's original code, the Euclidean `create_split` computes:
```
split_vector = p - q  (normalized)
offset_a = dot(split_vector, midpoint)  where midpoint = (p + q) / 2
margin = offset_a + dot(split_vector, query)
```

Our current implementation always uses `margin = dot(query, split_vector)`,
which assumes the hyperplane passes through the origin. This is correct for
angular/cosine distance (where vectors are direction-only) but WRONG for
L2/L1 where position matters.

**Impact:** L2-indexed annoy tables currently produce poor splits because the
hyperplane is not correctly positioned between the two cluster centroids.
Cosine works fine because the `two_means_split` already uses cosine distance
and normalization when `use_cosine=1`.

**Approach:**
1. For L2/L1: compute offset `a = -dot(split_vector, midpoint)` during build
2. Store the offset in the split node blob:
   ```
   [4 bytes: left_id] [4 bytes: right_id] [4 bytes: offset_a] [N bytes: split_vector]
   ```
3. During search: `margin = offset_a + dot(query, split_vector)`
4. This matches annoy's `Euclidean::create_split` and `Euclidean::margin`

**Notes:**
- The `annoy_encode_split_node` / `annoy_decode_split_node` functions need a
  flag or extra field for the offset
- The PQ distance computation (`min(d, margin)` / `min(d, -margin)`) is the
  same for all metrics — only the margin formula changes
- Unit tests should verify L2-indexed annoy tables produce correct results
- Currently no tests exist for `distance_metric=L2 INDEXED BY annoy()` — need
  to add them

---

## Priority 10: Imbalanced split retry (matching annoy)

**Problem:** The original annoy retries the split up to 3 times if the
partition is >95% imbalanced (one side gets >95% of items). Our implementation
only does one attempt and falls back to random assignment.

**Impact:** In high dimensions, random hyperplanes can produce extremely
imbalanced splits by chance. One bad split near the root creates a lopsided
tree where one subtree has most items, degrading search performance. Annoy's
3-retry heuristic significantly reduces the probability of bad splits.

**Approach:**
```c
for (int attempt = 0; attempt < 3; attempt++) {
    annoy_two_means_split(...);
    if (n_left > n_items * 0.05 && n_right > n_items * 0.05) {
        break;  // balanced enough
    }
    // retry with different random seed
}
// if still imbalanced after 3 attempts, random assign (current behavior)
```

**Notes:**
- Simple to implement, no schema changes
- May improve recall by 1-3% at no query-time cost
- Slightly increases build time (up to 3x for pathological data)

---

## Priority 11: Metadata filter oversampling

**Problem:** When KNN queries include metadata filters (e.g., `AND category =
'science'`), the annoy search returns candidates that may not match the filter.
If the filter is selective (e.g., only 5% of vectors match), many candidates
are wasted and the effective recall drops.

**Current behavior:** The annoy search returns exactly k results after
re-ranking. If metadata filters then remove some, the user gets fewer than k
results.

**Approach:** When metadata filters are present, automatically oversample:
collect `oversample * k` candidates from the annoy search, apply metadata
filters, then take top k from the filtered set.

The `oversample` factor could be:
- Fixed at e.g., 4x (simple, works for moderate selectivity)
- Adaptive based on filter selectivity (more complex, better for edge cases)
- User-configurable via an `oversample` parameter in the annoy config

**Notes:**
- The existing vec0 metadata filtering infrastructure already works with the
  KNN result set — we just need to provide more candidates
- This is the same pattern used by IVF's `oversample` parameter
- The annoy search function already supports returning more than k candidates
  (just increase the `k` parameter to the search function)
- Need to thread the oversample factor through the query path:
  `vec0BestIndex → idxStr → vec0Filter_knn_annoy → annoy_search_with_buffer`

---

## Priority 12: Extract annoy code to separate file

**Problem:** All annoy code is currently inline in `sqlite-vec.c`, which is
already 12000+ lines. The ANNOY_PLAN.md specified a separate `sqlite-vec-annoy.c`
file `#include`d into `sqlite-vec.c` (same pattern the IVF branch uses with
`sqlite-vec-ivf.c`).

**Approach:**
1. Move all annoy functions (everything between the `// Annoy` section markers)
   to a new `sqlite-vec-annoy.c` file
2. Add `#include "sqlite-vec-annoy.c"` in `sqlite-vec.c` at the appropriate point
3. This keeps the main file manageable and makes the annoy code self-contained
4. No behavioral changes — pure refactor

**Notes:**
- The IVF branch already demonstrates this pattern
- Forward declarations may be needed for functions in sqlite-vec.c that annoy
  calls (e.g., `diskann_distance_full`, `vector_column_byte_size`)
- The internal test header would need to declare annoy functions from the new file

---

## Priority 13: Dot product distance support

**Problem:** Annoy's original library supports dot product (inner product)
distance via a clever transformation. The current implementation only handles
cosine and L2/L1 correctly.

For dot product, annoy uses the Microsoft Research method:
1. Preprocess: compute global max norm across all vectors
2. Transform each vector `v` to `[v, sqrt(max_norm^2 - ||v||^2)]` (append one dim)
3. Now inner product similarity ≈ angular distance in the augmented space
4. Build trees in the augmented space using the angular algorithm

**Approach:**
- During `annoy_build_all()`, if distance_metric is dot product:
  1. Compute max norm across all vectors
  2. Augment each vector with the extra dimension
  3. Build trees with `use_cosine=1` on the augmented vectors
  4. Store augmented split vectors in nodes
- During `annoy_search()`:
  1. Augment the query vector the same way
  2. Run the normal angular search
  3. Re-rank with true dot product distance (not augmented)

**Notes:**
- sqlite-vec already has dot product distance for brute-force queries
- The augmentation adds 1 dimension (769 instead of 768) — negligible overhead
- The max norm needs to be stored in metadata (changes between rebuilds)
- This is a lower priority because cosine and L2 cover most use cases

---

## Non-priority items (nice to have)

- [ ] `EXPLAIN` support: show annoy search plan details in EXPLAIN output
- [ ] `vec0_debug()` output: include annoy index stats (n_trees, n_nodes,
      buffer size, tree depth, avg descendants per leaf)
- [ ] Concurrent query support: verify thread safety of cached prepared
      statements and json_each batch queries
- [ ] On-disk build: for datasets larger than memory, stream vectors through
      tree construction without loading all into RAM. Annoy's original lib
      supports `on_disk_build()` via mmap — we could write vectors to a temp
      file and mmap it during build
- [ ] Multiple vector columns with different index types (e.g., one annoy,
      one diskann) — parsing supports this but hasn't been tested
- [ ] RoaringBitmap for descendants: for large leaf buckets, a compressed
      bitmap is more space-efficient than packed i64 arrays. The ANNOY_PLAN
      mentioned this. At 48-item leaves it's not worth it, but if we increase
      MAX_DESCENDANTS it could help.
- [ ] Deterministic vs random seed: currently uses seed=42 for reproducibility.
      Should add a `seed` parameter to the annoy config for users who want
      different random forests, or use a hash of the table name.
- [ ] `get_nns_by_item()` equivalent: query by an existing item's rowid
      without providing a vector. Lookup the vector from `_annoy_vectors`
      and run the search. Useful for "find similar items" queries.
- [ ] int8 vector column support: the annoy index currently only works with
      float32 vectors. int8 vectors would need the two-means split and margin
      computation adapted to int8 arithmetic, or promote to float32 for
      tree operations and use int8 only for storage/re-ranking.
- [ ] Annoy index statistics command: `INSERT INTO t(rowid, embedding) VALUES
      ('stats', NULL)` that prints tree depth, node count, leaf sizes, buffer
      size, etc. Useful for diagnosing performance issues.
- [ ] Build progress reporting: for large datasets, the build takes minutes.
      Currently silent. Could use `sqlite3_progress_handler()` or print
      progress to the SQLite error log.
