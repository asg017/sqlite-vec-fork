# IVF Optimization TODO

## Completed

- [x] **Skip chunk writes for IVF columns** — eliminated 52% of insert time and 2x DB size
- [x] **SIMD distance function** — NEON cosine_float_neon() for queries, centroids, assignment
- [x] **Use column's distance_metric** — IVF now uses cosine/L2/L1 based on column config
- [x] **Batch cell reads** — single `WHERE centroid_id IN (...)` instead of N queries
- [x] **Runtime nprobe** — `INSERT INTO t(id) VALUES ('nprobe=N')` without rebuild
- [x] **IVF quantization** — quantizer=int8/binary with oversample re-ranking
- [ ] **Runtime oversample** — `INSERT INTO t(id) VALUES ('oversample=N')` without rebuild (same pattern as nprobe)
- [x] **Fixed-size cells (64 vectors)** — avoids overflow page traversal (110x insert speedup)
- [x] **Cached prepared statements** — hot-path stmts cached in vec0_vtab

## Current Benchmarks

### 100k (nlist=316, train=16x)
```
nprobe  qry(ms)  recall
     8     4.8   0.934    ← 7x faster than flat (34ms)
    16     8.2   0.968
    32    15.2   0.992
flat      34.2   1.000
bit(8)    19.7   0.884
int8(4)   19.5   0.996
```

### 1M (nlist=1000, train=16x)
```
nprobe  qry(ms)  recall
     8    98.6   0.890    ← tied with bit(8) 96ms/0.918
    16   161.1   0.950
    32   212.1   0.980
flat     351.0   1.000
bit(8)    96.2   0.918
int8(4)  254.5   0.994
```

## Insert Bottlenecks (1M, nlist=1000)

Insert: **2000 rows/s** (vs 50k for flat). Degrades over time.

- [ ] **Cache centroids in memory** — each insert re-reads all 1000 centroids
      from SQLite via `stmtIvfCentroidsAll`. At 768-dim float32, that's 1000×3KB =
      3MB per insert read from B-tree. Should load once into a float array in
      vec0_vtab, invalidate on compute-centroids/set-centroid/clear-centroids.
      **Expected: 3-5x insert speedup.**

- [ ] **Batch blob writes** — 3 `sqlite3_blob_open`/write/close per insert
      (validity, rowids, vectors). Keep handles open for the current cell,
      close when cell fills (every 64 inserts) or on xSync.

- [ ] **Skip chunk position finding for IVF-only tables** — still runs
      `vec0Update_InsertNextAvailableStep` even though chunk data isn't used.
      Writes validity/rowid to chunks. Could skip entirely if all vector
      columns use IVF.

## Query Bottlenecks (1M, nlist=1000, nprobe=16 → 161ms)

- [ ] **Disk I/O dominates at 1M** — cell blobs don't fit in page cache.
      Each probed centroid has ~1000 vectors across ~16 cell rows.
      16 centroids × 16 cells × 200KB = ~50MB of blob reads per query.
      At 100k this fits in cache (fast); at 1M it doesn't (slow).

- [ ] **Top-k heap instead of qsort** — at nprobe=16, scanning ~16k vectors
      and sorting all of them. A min-heap of size k=10 would avoid the sort.
      Small win (~9% at 100k) but grows with nprobe.

- [ ] **SIMD k-means** — training on 16k vectors with nlist=1000 takes 130s.
      The inner loop uses `ivf_l2_dist` (scalar) not `distance_l2_sqr_float`
      (NEON). Switching would cut training time ~4x.

## Recall Improvement

- [ ] **Train on more vectors** — 16x nlist gives decent recall (0.95 at
      nprobe=16) but 32-64x would tighten clusters. Limited by k-means speed.
      SIMD k-means would make this practical.

- [ ] **IVF-PQ** — product quantization within cells would compress vectors
      and make cell scans faster (compute distance on compressed codes).
      Major feature, not a quick optimization.
