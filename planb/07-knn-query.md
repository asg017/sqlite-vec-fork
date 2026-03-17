# Step 7: KNN Query Path

*Reference: `plans/diskann-plans/10-knn-query-integration.md` for the DiskANN equivalent.*

## Overview

The KNN query has three modes depending on index state:
1. **Flat mode** — no centroids, brute-force scan everything.
2. **Hybrid mode** — centroids exist but unassigned vectors remain. IVF probe +
   brute-force scan of unassigned, merge results.
3. **Pure IVF mode** — all vectors assigned. Probe `nprobe` cells only.

## Query Flow

### `vec0Filter_knn_ivf()`

Dispatched from `vec0Filter_knn()` when the vector column has `ivf.enabled`:

```c
static int vec0Filter_knn_ivf(
  vec0_vtab *p,
  vec0_cursor *cursor,
  int vectorColIdx,
  const void *queryVector,
  int queryVectorSize,
  int k
) {
  struct Vec0IvfConfig *cfg = &p->vector_columns[vectorColIdx].ivf;

  if (!ivf_is_trained(p, vectorColIdx)) {
    // === FLAT MODE ===
    return ivf_query_flat(p, cursor, vectorColIdx, queryVector, queryVectorSize, k);
  }

  // === IVF MODE (pure or hybrid) ===

  // Step 1: Find top nprobe centroids
  int nprobe = cfg->nprobe;
  int *probe_ids = sqlite3_malloc(nprobe * sizeof(int));
  ivf_find_top_centroids(p, vectorColIdx, queryVector, queryVectorSize,
                         nprobe, probe_ids);

  // Step 2: Scan vectors in those cells
  // Accumulate (rowid, distance) pairs
  struct IvfCandidate *candidates = NULL;
  int nCandidates = 0;

  for (int i = 0; i < nprobe; i++) {
    ivf_scan_cell(p, vectorColIdx, probe_ids[i], queryVector,
                  &candidates, &nCandidates);
  }

  // Step 3: If unassigned vectors exist, scan those too (hybrid mode)
  int nUnassigned = ivf_count_unassigned(p, vectorColIdx);
  if (nUnassigned > 0) {
    ivf_scan_unassigned(p, vectorColIdx, queryVector,
                        &candidates, &nCandidates);
  }

  // Step 4: Sort by distance, take top k
  qsort(candidates, nCandidates, sizeof(struct IvfCandidate), cmp_distance);
  int nResults = nCandidates < k ? nCandidates : k;

  // Step 5: Populate cursor knn_data
  cursor->knn_data->nResults = nResults;
  for (int i = 0; i < nResults; i++) {
    cursor->knn_data->rowids[i] = candidates[i].rowid;
    cursor->knn_data->distances[i] = candidates[i].distance;
  }

  sqlite3_free(probe_ids);
  sqlite3_free(candidates);
  return SQLITE_OK;
}
```

### `ivf_find_top_centroids()`

Find the `nprobe` closest centroids to the query vector:

```c
static int ivf_find_top_centroids(
  vec0_vtab *p,
  int colIdx,
  const void *query,
  int querySize,
  int nprobe,
  int *out_centroid_ids  // output: nprobe centroid IDs
) {
  // Read all centroids, compute distances, partial sort for top nprobe
  // For small nlist (<1000), a simple full sort is fine.
  // For larger nlist, use a min-heap or partial quickselect.

  struct { int id; float dist; } *centroid_dists;
  int nlist = 0;

  // SELECT centroid_id, centroid FROM _ivf_centroids
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    centroid_dists[nlist].id = sqlite3_column_int(stmt, 0);
    const void *c = sqlite3_column_blob(stmt, 1);
    centroid_dists[nlist].dist = compute_distance(query, c, D,
      p->vector_columns[colIdx].distance_metric);
    nlist++;
  }

  // Sort by distance, take first nprobe
  qsort(centroid_dists, nlist, sizeof(*centroid_dists), cmp_dist);
  for (int i = 0; i < nprobe && i < nlist; i++) {
    out_centroid_ids[i] = centroid_dists[i].id;
  }

  return SQLITE_OK;
}
```

### `ivf_scan_cell()`

Scan all vectors in a single centroid cell:

```c
static int ivf_scan_cell(
  vec0_vtab *p,
  int colIdx,
  int centroid_id,
  const void *query,
  struct IvfCandidate **candidates,
  int *nCandidates
) {
  // SELECT rowid, vector FROM _ivf_vectors WHERE centroid_id = ?
  // For each row, compute distance to query, append to candidates
  return SQLITE_OK;
}
```

The `centroid_id` index on `_ivf_vectors` makes this an efficient index scan.

### `ivf_query_flat()`

Brute-force scan for flat mode:

```c
static int ivf_query_flat(
  vec0_vtab *p,
  vec0_cursor *cursor,
  int colIdx,
  const void *query,
  int querySize,
  int k
) {
  // Scan _ivf_unassigned (all vectors are here in flat mode)
  // SELECT rowid, vector FROM _ivf_unassigned
  // Compute distances, sort, take top k
  // Same as existing chunk-based brute force, just different source table
  return SQLITE_OK;
}
```

## xBestIndex Integration

In `vec0BestIndex()`, when a vector column has IVF enabled:

- **Flat mode (not trained):** Cost estimate same as brute-force (~N comparisons).
- **Trained mode:** Cost estimate = `nprobe * (N / nlist)` + overhead.
  This is significantly lower than brute-force, which tells SQLite to prefer
  this plan.

```c
if (col->ivf.enabled) {
  if (ivf_is_trained(p, colIdx)) {
    // Approximate cost: probe nprobe cells out of nlist
    pIdxInfo->estimatedCost = (double)(cfg->nprobe) / cfg->nlist * N;
  } else {
    // Flat mode: full scan
    pIdxInfo->estimatedCost = (double)N;
  }
}
```

## Query-Time nprobe Override (Future Enhancement)

Allow overriding `nprobe` at query time via a special constraint:

```sql
SELECT id, distance
FROM vec_items
WHERE contents_embedding MATCH :query
  AND k = 20
  AND nprobe = 32;  -- override default
```

This requires adding `nprobe` as a hidden column and handling it in
xBestIndex/xFilter. Can be deferred to a later step.

## Candidate Struct

```c
struct IvfCandidate {
  i64 rowid;
  float distance;
};
```

## Performance Characteristics

For N=1M vectors, D=1024, nlist=256, nprobe=16:
- **Centroid scan:** 256 distance computations (fast, ~0.2ms)
- **Cell scan:** 16 * (1M/256) = ~62,500 distance computations
- **vs. brute force:** 1,000,000 distance computations
- **Speedup:** ~16x with nprobe=16/nlist=256

Recall is typically 90%+ with nprobe/nlist >= 0.05.

## Files Changed

- `sqlite-vec-ivf.c`: Add `vec0Filter_knn_ivf()`, `ivf_find_top_centroids()`,
  `ivf_scan_cell()`, `ivf_scan_unassigned()`, `ivf_query_flat()`,
  `ivf_bestindex_cost()`.
- `sqlite-vec.c`: ~3-line dispatch in `vec0Filter_knn()` to call
  `vec0Filter_knn_ivf()` when `ivf.enabled`. ~3-line cost override in
  `vec0BestIndex()` calling `ivf_bestindex_cost()`.

## Python Tests

```python
def test_knn_flat_mode(db):
    """Before training, KNN does brute-force and returns correct results."""
    db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[4] indexed by ivf(nlist=4))")
    db.execute("INSERT INTO t(rowid, v) VALUES (1, ?)", [_f32([1,0,0,0])])
    db.execute("INSERT INTO t(rowid, v) VALUES (2, ?)", [_f32([2,0,0,0])])
    db.execute("INSERT INTO t(rowid, v) VALUES (3, ?)", [_f32([9,0,0,0])])

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE v MATCH ? AND k=2",
        [_f32([1.5, 0, 0, 0])]
    ).fetchall()
    assert rows[0][0] in (1, 2)  # closest vectors
    assert rows[1][0] in (1, 2)

def test_knn_after_training(db):
    """After training, KNN uses IVF probe and returns correct results."""
    db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[4] indexed by ivf(nlist=4, nprobe=2))")
    for i in range(100):
        db.execute("INSERT INTO t(rowid, v) VALUES (?, ?)", [i, _f32([i,0,0,0])])

    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE v MATCH ? AND k=5",
        [_f32([50.0, 0, 0, 0])]
    ).fetchall()
    assert len(rows) == 5
    # Top result should be rowid=50 (exact match)
    assert rows[0][0] == 50
    assert rows[0][1] < 0.01

def test_knn_hybrid_mode(db):
    """Vectors inserted after training are still found via hybrid scan."""
    db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[4] indexed by ivf(nlist=4, nprobe=4))")
    for i in range(40):
        db.execute("INSERT INTO t(rowid, v) VALUES (?, ?)", [i, _f32([i,0,0,0])])

    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")

    # Insert new vector after training — goes to assigned table
    db.execute("INSERT INTO t(rowid, v) VALUES (999, ?)", [_f32([20.5, 0, 0, 0])])

    rows = db.execute(
        "SELECT rowid FROM t WHERE v MATCH ? AND k=1",
        [_f32([20.5, 0, 0, 0])]
    ).fetchall()
    assert rows[0][0] == 999  # newly inserted vector is found
```
