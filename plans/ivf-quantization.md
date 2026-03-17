# IVF Quantization Plan

## Goal

Add quantized IVF indexing: store and compare centroids/cell vectors in a
compressed representation (int8 or binary) while keeping full-precision vectors
available for optional re-scoring. This targets the two biggest query-time
bottlenecks: distance computation (49%) and I/O (41%).

## Proposed API

```sql
-- Binary quantization (32x storage reduction for cells)
CREATE VIRTUAL TABLE v USING vec0(
  embedding float[768] indexed by ivf(nlist=256, nprobe=16, quantizer=binary)
);

-- Int8 scalar quantization (4x storage reduction for cells)
CREATE VIRTUAL TABLE v USING vec0(
  embedding float[768] indexed by ivf(nlist=256, nprobe=16, quantizer=int8)
);

-- Oversample + re-score: fetch 10x candidates with quantized distances,
-- then re-rank with full-precision vectors
CREATE VIRTUAL TABLE v USING vec0(
  embedding float[768] indexed by ivf(nlist=256, nprobe=16, quantizer=binary, oversample=10)
);
```

### Parameter semantics

| Parameter | Values | Default | Meaning |
|-----------|--------|---------|---------|
| `quantizer` | `none`, `int8`, `binary` | `none` | How cell vectors + centroids are stored/compared |
| `oversample` | integer >= 1 | 1 (no oversampling) | Fetch `oversample * k` candidates via quantized distance, then re-score top k with full vectors |

### Interaction with `nlist`

`nlist` and `quantizer` are orthogonal:
- `nlist` controls **how many clusters** (coarse partitioning granularity)
- `quantizer` controls **how vectors within cells are stored and compared**

They compose naturally: `nlist=256, quantizer=binary` means 256 clusters where
each cluster's vectors are stored as bit vectors. Increasing `nlist` reduces the
number of vectors scanned per query; quantization makes each scan cheaper.

At 1M vectors with 768 dims:
- `nlist=256, quantizer=none`: scan ~4k vectors × 3072 bytes = 12 MB per query
- `nlist=256, quantizer=binary`: scan ~4k vectors × 96 bytes = 375 KB per query
- `nlist=256, quantizer=int8`: scan ~4k vectors × 768 bytes = 3 MB per query

So quantization gives 4-32x I/O reduction **within** each probed cell,
while `nlist` controls how many cells are probed. Both levers matter.

### Oversample rationale

Quantized distances are approximate. `oversample=N` compensates:
1. Query with quantized distances, collect top `N * k` candidates
2. Look up each candidate's full-precision vector
3. Re-score with exact distance, return top k

This is the standard IVF-PQ/IVF-SQ re-ranking pattern. Without oversample,
binary quantization recall drops significantly (hamming distance is a rough
proxy for L2/cosine). With oversample=10, recall approaches brute-force while
keeping the fast quantized scan as the first pass.

Oversample requires storing full-precision vectors somewhere accessible by
rowid. See "Full-vector storage" below.

## Architecture

### Current state

```
User vector (float32)
  → find nearest centroid (float32 centroids, SIMD distance)
  → store in cell (float32 packed blob, 64 vectors per cell)
  → query: scan cell blobs, compute float32 distances
```

### Proposed state

```
User vector (float32)
  → quantize to target type
  → find nearest centroid (quantized centroids, type-appropriate distance)
  → store quantized vector in cell
  → if oversample > 1: also store full vector in KV table
  → query:
      1. scan cell blobs with quantized distance (fast, small I/O)
      2. if oversample: fetch full vectors for top N*k, re-score, return top k
```

## Implementation plan

### Phase 1: Config parsing and plumbing

**Files**: `sqlite-vec-ivf.c` (parser), `sqlite-vec.c` (struct)

1. Extend `Vec0IvfConfig`:
   ```c
   struct Vec0IvfConfig {
     int enabled;
     int nlist;
     int nprobe;
     int quantizer;    // VEC0_IVF_QUANTIZER_NONE / _INT8 / _BINARY
     int oversample;   // >= 1, default 1
   };
   ```

2. Extend `vec0_parse_ivf_options()` to accept string-valued params:
   - `quantizer=binary` → `VEC0_IVF_QUANTIZER_BINARY`
   - `quantizer=int8` → `VEC0_IVF_QUANTIZER_INT8`
   - `quantizer=none` → `VEC0_IVF_QUANTIZER_NONE` (default)
   - `oversample=N` → integer N >= 1
   - Parser currently only handles `TOKEN_TYPE_DIGIT` for values.
     Need to also accept `TOKEN_TYPE_IDENTIFIER` for string values like `binary`.

3. Validation rules:
   - `oversample > 1` requires `quantizer != none` (error otherwise — no point
     oversampling when already using full precision)
   - `quantizer=binary` only valid for float32 columns (binary quantization of
     int8 columns is technically possible but weird)
   - Column must be float32 element type (quantization is a storage optimization
     on top of float32 source vectors)

### Phase 2: Shadow table changes

**Files**: `sqlite-vec-ivf.c`

#### Centroid table — unchanged schema, different content

The `_ivf_centroids` table schema stays the same (`centroid_id INTEGER PRIMARY KEY, centroid BLOB`). What changes is the **content** of the centroid blob:

- `quantizer=none`: centroid is `D * 4` bytes (float32), same as today
- `quantizer=int8`: centroid is `D` bytes (int8)
- `quantizer=binary`: centroid is `D / 8` bytes (bit-packed)

The centroid representation matches the quantizer so that centroid distance
computation uses the same fast path as cell scanning.

**Design decision**: Should centroids be quantized too, or kept as float32?

Option A: **Quantize centroids** (proposed). Centroid-to-query distance uses the
same quantized distance function. Simpler, consistent. Coarse quantizer error is
acceptable because we probe `nprobe` centroids anyway — the top-1 centroid
assignment might shift but top-`nprobe` is robust.

Option B: **Keep centroids as float32**. More accurate coarse assignment, but
requires two distance functions (float32 for centroids, quantized for cells) and
doesn't save much — centroid storage is `nlist * D * 4` bytes, tiny vs cells.

**Recommendation**: Option A. Simplicity wins and nprobe compensates. If recall
is poor with binary centroids, we can revisit. Int8 centroids should be fine.

#### Cell table — quantized vector blobs

```sql
CREATE TABLE _ivf_cells00 (
  centroid_id INTEGER NOT NULL,
  n_vectors INTEGER NOT NULL DEFAULT 0,
  validity BLOB NOT NULL,
  rowids BLOB NOT NULL,
  vectors BLOB NOT NULL   -- now quantized: D/8 bytes (binary) or D bytes (int8) per slot
)
```

Schema is identical; blob sizes change:

| quantizer | bytes per vector | cell blob size (64 slots, 768-dim) |
|-----------|-----------------|-------------------------------------|
| none | 3072 | 192 KB |
| int8 | 768 | 48 KB |
| binary | 96 | 6 KB |

`VEC0_IVF_CELL_MAX_VECTORS` can stay at 64 — cells get much smaller with
quantization, which is the point.

#### Full-vector KV table (new, only when oversample > 1)

```sql
CREATE TABLE _ivf_vectors00 (
  rowid INTEGER PRIMARY KEY,
  vector BLOB NOT NULL   -- full float32 vector
)
```

Simple rowid-keyed store for full-precision vectors. Used only during re-scoring.
No blob-packing tricks — one row per vector, direct lookup by rowid.

**Why not reuse chunks?** Chunks are designed for sequential scan (dense packing,
positional addressing). Re-scoring needs random access by rowid. A simple KV
table with `rowid PRIMARY KEY` (true integer primary key = rowid alias) gives
O(1) lookup via B-tree. No SHADOW_TABLE_ROWID_QUIRK here since we use
`INTEGER PRIMARY KEY` properly.

**Alternative considered**: Store full vectors in the existing cell blobs alongside
quantized vectors. Rejected — doubles cell blob size, defeats the purpose of
quantization for the scan phase, and complicates the cell layout.

**When oversample=1**: This table is not created. No full vectors stored. The
quantized distances are the final answer. This is the "fast and approximate" mode.

### Phase 3: Quantization helpers

**Files**: `sqlite-vec-ivf.c` (or new `sqlite-vec-ivf-quantize.c`)

Need internal functions to quantize float32 vectors to int8/binary. These are
similar to the existing `vec_quantize_int8()` and `vec_quantize_binary()` SQL
functions but as C helpers operating on raw float arrays:

```c
// Quantize float32 vector to int8 (assumes pre-normalized to [-1,1])
// Output: D bytes
static void ivf_quantize_int8(const float *src, int8_t *dst, int D);

// Quantize float32 vector to binary (sign-bit quantization)
// Output: D/8 bytes
static void ivf_quantize_binary(const float *src, uint8_t *dst, int D);

// Size of a quantized vector in bytes
static int ivf_quantized_vec_size(int D, int quantizer);
```

The int8 quantization needs normalization bounds. Two options:
1. **Global min/max** computed during training (stored in `_info` table)
2. **Assume unit-normalized** input (common for embedding models)

Recommend option 2 initially (same as `vec_quantize_int8("unit")`), with option
1 as a future enhancement.

### Phase 4: K-means for quantized types

**Files**: `sqlite-vec-ivf-kmeans.c`

Current k-means operates purely on float32. Two approaches:

#### Approach A: Quantize-then-cluster (simpler, recommended first)

1. Receive training vectors as float32
2. Quantize all training vectors to target type
3. Run k-means in quantized space:
   - For int8: k-means on int8 vectors using `distance_l2_sqr_int8()`
   - For binary: k-means on bit vectors using `distance_hamming()`
4. Output quantized centroids

Requires new k-means variants or a generic version parameterized by type and
distance function.

**Problem with binary k-means**: The "mean" of a set of bit vectors isn't well
defined. Standard approach: compute mean in float32 space, then quantize to bits.
This means the update step works in float32, but assignment step uses hamming
distance.

Proposed k-means for binary quantizer:
```
1. Quantize all training vectors to bits
2. Keep float32 copies for centroid updates
3. Loop:
   a. Assignment: hamming distance between bit-vectors and bit-centroids
   b. Update: average assigned float32 vectors, then quantize to bits
```

For int8 quantizer, standard k-means works directly on int8 with L2 distance.
The centroid update computes mean in int32 accumulator, rounds to int8.

#### Approach B: Cluster-in-float32-then-quantize (alternative)

1. Run standard float32 k-means (existing code)
2. Quantize the resulting centroids
3. Store and compare everything in quantized space going forward

Simpler but the centroids were optimized for float32 distances, not quantized
distances. May cause suboptimal cluster boundaries. Could be acceptable for int8
(small distortion) but poor for binary (large distortion).

**Recommendation**: Approach A for correctness, with the binary k-means hybrid
described above. Approach B as a quick MVP.

### Phase 5: Insert path changes

**Files**: `sqlite-vec-ivf.c`

#### `ivf_insert()` modifications:

```
1. Receive float32 vector from user (always — the column type doesn't change)
2. Quantize to target type: ivf_quantize_{int8,binary}()
3. Find nearest centroid using quantized distance
4. Insert quantized vector into cell blob
5. If oversample > 1: INSERT full float32 vector into _ivf_vectors table
6. Insert into rowid_map (unchanged)
```

Key change: `ivf_cell_insert()` writes a smaller blob per vector. The validity
and rowids blobs are unchanged; only the vectors blob shrinks.

`ivf_vec_size()` needs to return the quantized size:
```c
static int ivf_vec_size(vec0_vtab *p, int col_idx) {
  int D = p->vector_columns[col_idx].dimensions;
  switch (p->vector_columns[col_idx].ivf.quantizer) {
    case VEC0_IVF_QUANTIZER_INT8:    return D;
    case VEC0_IVF_QUANTIZER_BINARY:  return D / 8;
    default:                          return D * sizeof(float);
  }
}
```

### Phase 6: Query path changes

**Files**: `sqlite-vec-ivf.c`

#### Without oversample (oversample=1):

```
1. Quantize query vector
2. Score centroids using quantized distance → top nprobe
3. Scan cells: quantized distance between quantized query and quantized cell vectors
4. Sort candidates, return top k
```

Distance dispatch in `ivf_distance()` and `ivf_find_nearest_centroid()` needs
to handle all three types. Rather than one mega-switch, use function pointers
set once at table open:

```c
// In vec0_vtab or Vec0IvfConfig:
float (*ivf_distance_fn)(const void *a, const void *b, int D);
int (*ivf_nearest_centroid_fn)(const void *vec, const void *centroids, int D, int k);
```

For binary: distance is hamming (integer), cast to float for uniform sorting.

#### With oversample (oversample > 1):

```
1. Quantize query vector
2. Score centroids → top nprobe
3. Scan cells with quantized distance → collect top (oversample * k) candidates
   (need a top-k heap here, not full sort — good time to implement)
4. For each of the top (oversample * k) candidates:
   a. Look up full vector from _ivf_vectors by rowid
   b. Compute exact float32 distance
5. Sort re-scored candidates, return top k
```

The re-scoring step is `oversample * k` random reads from `_ivf_vectors`.
At oversample=10, k=10: 100 lookups. Each is a B-tree probe, very fast.

### Phase 7: compute-centroids command changes

**Files**: `sqlite-vec-ivf.c`

`ivf_cmd_compute_centroids()` currently:
1. Loads all float32 vectors from cells
2. Runs float32 k-means
3. Saves float32 centroids
4. Redistributes float32 vectors into cells

With quantization:
1. Load all vectors from cells (they're quantized if table already has data,
   or float32 if this is first training on unassigned data)
   - For unassigned data: vectors were quantized on insert, stored as quantized
   - Need to handle the case where original float32 is needed for binary k-means update step
   - **Solution**: If oversample > 1, full vectors are in `_ivf_vectors` — use those.
     If oversample=1, we only have quantized vectors. Binary k-means update step
     would need to approximate (e.g., use hamming centroid = majority vote per bit).
2. Run quantized k-means (Phase 4)
3. Save quantized centroids
4. Redistribute quantized vectors into cells

**Complication**: For `quantizer=binary, oversample=1`, the k-means update step
can't compute float32 means (no float32 vectors available). Options:
- Majority-vote centroids (each bit = majority of assigned vectors' bits). This
  is the standard binary k-means approach. Works but quality may suffer.
- Require oversample > 1 for binary quantizer. Simpler, and binary without
  re-scoring has poor recall anyway. **Recommended**.
- Store float32 vectors temporarily during training, discard after.

### Phase 8: Delete path changes

**Files**: `sqlite-vec-ivf.c`

Mostly unchanged. `ivf_delete()` already uses rowid_map to find cell + slot,
then zeroes the validity bit. The vector blob slot is just dead space until
the cell is compacted.

Additional work when `oversample > 1`: also DELETE from `_ivf_vectors`.

### Phase 9: Point query changes

`ivf_get_vector_data()` currently reads float32 from cell. With quantization,
the cell contains quantized data. For returning the original vector:

- If `oversample > 1`: read from `_ivf_vectors` (full precision)
- If `oversample = 1`: return the quantized vector? Or error?

**Decision**: Point queries should return the original float32 vector. This means:
- `oversample > 1`: straightforward, read from KV table
- `oversample = 1, quantizer != none`: The original vector is lost. Point query
  would return the quantized approximation, which may surprise users.

**Recommendation**: Always store full vectors in `_ivf_vectors` when
`quantizer != none`, regardless of oversample. The oversample parameter only
controls whether re-scoring happens during KNN queries. This way:
- Point queries always return full precision
- Re-scoring is available if oversample > 1
- Storage cost: full vectors always stored, but cell scan uses quantized

This simplifies the design: `_ivf_vectors` exists whenever `quantizer != none`.
The **only** thing `oversample` controls is the query re-ranking behavior.

## Implementation order

1. **Config parsing** (Phase 1) — parser changes, struct changes, validation
2. **Quantization helpers** (Phase 3) — pure C, testable independently
3. **Shadow tables** (Phase 2) — create `_ivf_vectors`, adjust cell blob sizes
4. **Insert path** (Phase 5) — quantize on insert, dual-write to cells + KV
5. **K-means** (Phase 4) — start with Approach B (cluster float32, quantize after), upgrade to A later
6. **Query path** (Phase 6) — quantized scan, oversample re-ranking
7. **compute-centroids** (Phase 7) — integrate quantized k-means
8. **Delete + point query** (Phase 8, 9) — cleanup paths

Each step is independently testable. Steps 1-4 can land without changing query
behavior (just store quantized, query still works on whatever's in the cells).

## Testing strategy

New test file: `tests/test-ivf-quantization.py`

1. **Smoke tests**: Create tables with each quantizer, insert vectors, compute
   centroids, run KNN queries. Verify results returned (not necessarily recall).

2. **Recall tests**: Compare recall at various configurations:
   - `quantizer=none` (baseline)
   - `quantizer=int8, oversample=1`
   - `quantizer=int8, oversample=10`
   - `quantizer=binary, oversample=10`
   - `quantizer=binary, oversample=50`

3. **Round-trip tests**: Insert vectors, verify point queries return original
   float32 vectors (not quantized approximations).

4. **Edge cases**: Empty table, single vector, nlist > n_vectors, delete + re-insert.

5. **Benchmark** (in `ivf-benchmarks/bench_ivf_quantized.py`):
   Compare all approaches at 100k and 1M. Script should test:

   ```
   Baselines:
     flat                          — brute force
     int8(oversample=4)            — int8 quantized rescore
     bit(oversample=8)             — binary quantized rescore

   IVF (no quantization):
     ivf(nlist=sqrt(N), nprobe=8)  — current best IVF
     ivf(nlist=sqrt(N), nprobe=16)
     ivf(nlist=sqrt(N), nprobe=32)

   IVF + int8 quantization:
     ivf(nlist=sqrt(N), nprobe=8, quantizer=int8, oversample=1)
     ivf(nlist=sqrt(N), nprobe=8, quantizer=int8, oversample=4)
     ivf(nlist=sqrt(N), nprobe=16, quantizer=int8, oversample=4)

   IVF + binary quantization:
     ivf(nlist=sqrt(N), nprobe=16, quantizer=binary, oversample=10)
     ivf(nlist=sqrt(N), nprobe=32, quantizer=binary, oversample=10)
     ivf(nlist=sqrt(N), nprobe=32, quantizer=binary, oversample=50)
   ```

   Use runtime `nprobe=N` command to sweep nprobe without rebuilding.
   Training: 16*nlist vectors. Use `bench_nprobe_sweep.py` pattern.

## Expected performance characteristics

At 1M vectors, 768 dimensions, nlist=256, nprobe=16, k=10:

| Config | Cell scan I/O | Distance cost | Recall (est.) | Notes |
|--------|--------------|---------------|---------------|-------|
| quantizer=none | 12 MB | 49% of query | ~0.87 | Current |
| quantizer=int8, oversample=1 | 3 MB | ~30% (int8 SIMD) | ~0.83 | 4x I/O reduction |
| quantizer=int8, oversample=10 | 3 MB + 100 lookups | ~30% + rescore | ~0.86 | Near-baseline recall |
| quantizer=binary, oversample=1 | 375 KB | ~10% (hamming) | ~0.60 | Rough distances |
| quantizer=binary, oversample=10 | 375 KB + 100 lookups | ~10% + rescore | ~0.82 | Rescore recovers recall |
| quantizer=binary, oversample=50 | 375 KB + 500 lookups | ~10% + rescore | ~0.86 | More rescore = better |

The big win is binary + oversample: 32x less I/O for the scan phase, with
re-scoring recovering most recall. This directly attacks the 41% I/O bottleneck.

## Open questions

1. **Should `quantizer=binary` require `oversample > 1`?** Binary without
   re-scoring has poor recall. Could enforce this or just document it.

2. **Normalization for int8 quantization**: Assume unit-normalized inputs, or
   compute global min/max during training? Unit-normalized is simpler and covers
   most embedding models.

3. **Should centroids always stay float32?** The plan proposes quantizing them
   too, but keeping float32 centroids with quantized cells is a valid hybrid
   (accurate coarse assignment, fast cell scan). Could be a future `centroid_precision`
   option.

4. **Product quantization (PQ)**: Not in scope for this plan, but a natural
   next step. PQ subdivides vectors into subspaces and quantizes each
   independently, giving much better compression/recall tradeoffs than scalar
   quantization. Would be a separate `quantizer=pq` option.

5. **Training data source for oversample=1 binary**: If we always store full
   vectors in `_ivf_vectors` (per the Phase 9 recommendation), this is moot.
   But if we want a truly minimal-storage binary mode, we need the
   majority-vote k-means approach.
