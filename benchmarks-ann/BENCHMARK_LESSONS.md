# Benchmark Lessons and Analysis

Results from benchmarking all 5 index types (flat/rescore/IVF/DiskANN/annoy) across
10k, 100k, and 250k vectors on COHERE-medium-768d (cosine distance), Apple M1.

## Per-index learnings

### Rescore

Best overall value proposition. Near-zero build overhead (no training step), fast inserts
(70k rows/s), and strong query speedup via quantized coarse pass + float rescore.

- **rescore-bit-os8** is the speed champion: 4x faster than brute at 100k (8ms vs 33ms),
  but recall caps around 0.86-0.88. The bit quantization is too lossy for 768d cosine.
- **rescore-int8-os4** hits the sweet spot: 0.99 recall with 2x speedup over brute at 100k.
  The int8 quantizer preserves enough information for good candidate selection.
- **rescore-int8-os8** matches brute-force recall (0.998) but query time is nearly identical
  to brute at scale (50ms vs 84ms at 250k). The oversample=8 means scanning 80 candidates
  then rescoring, which approaches full scan cost.
- Profiling shows rescore spends 63% of CPU in `min_idx` (the top-k candidate selection),
  not distance computation. This is because the bit/int8 distance is cheap but the min-heap
  selection over all chunks is the bottleneck.

**Lesson**: int8 quantizer with oversample=4 is the practical default. Bit quantizer is
only useful when you can tolerate 0.86 recall for maximum speed.

### IVF

Best query latency at scale, but extremely expensive to build.

- At 100k: ivf-n128-p16 achieves 6.4ms queries (5x faster than brute) at 0.957 recall.
  This is the best latency of any index at this scale.
- Training time dominates build cost and scales super-linearly: nlist=64 takes 53s,
  nlist=128 takes 105s, nlist=256 takes 208s, nlist=512 takes 1071s at 250k.
- Insert throughput degrades from 27k rows/s at start to 2.2k rows/s by 250k. The IVF
  insert path does a nearest-centroid lookup per vector, which is O(nlist) per insert.
- At 250k, ivf-n512-p32 achieves 9.5ms (best query time overall) but takes 20 minutes
  to build. ivf-n128-p16 is 15ms and builds in 6 minutes.
- Profiling shows 57% pread (reading cell vectors from disk), 15% distance computation,
  12% qsort (sorting candidates). The scan is I/O-bound at scale.

**Lesson**: IVF is the best query performer but only practical for read-heavy workloads
that can afford a long training step. nlist=sqrt(N) and nprobe=nlist/8 is a good starting
point. The k-means training is the Achilles heel.

### DiskANN

Severely bottlenecked by per-row graph construction. Unusable at current implementation maturity.

- Insert throughput is 80-250 rows/s (400x slower than rescore/annoy). At 10k vectors,
  DiskANN takes 2+ minutes to build vs 0.1s for rescore.
- Query quality is reasonable (0.919 recall at 10k) but query latency (3.75ms) isn't
  competitive with IVF (1.34ms) or rescore (0.97ms).
- **DiskANN int8 quantizer is broken**: 0.009 recall at 10k. The quantized neighbor
  vectors are not producing meaningful distance approximations. This is a bug.
- DB size is large (118MB for 10k with binary, 586MB with int8) due to per-node graph
  storage overhead.

**Lesson**: DiskANN needs O(1) amortized insert (batched graph updates) to be competitive.
The int8 quantizer needs debugging. Skip DiskANN for now; it's research-grade.

### Annoy

Fast inserts but poor query scaling. Storage blowup is the critical issue.

- Insert throughput is the fastest (100k+ rows/s) because annoy just appends to buffer
  tables with no per-row index maintenance.
- Build (tree construction) scales linearly with n_trees: ~1.5s/tree at 10k, ~1.7s/tree
  at 100k. At 100k with 50 trees, build takes 85s.
- **Query latency scales terribly**: annoy-t50 goes from 6.2ms at 10k to 55ms at 100k
  (worse than brute-force at 33ms!). This is because each query traverses 50 trees, each
  requiring multiple SQLite B-tree lookups per node.
- **Storage is enormous**: 50 trees at 100k = 1.6GB (5.4x brute-force). Each tree node
  is a separate row in a SQLite table, and with 768-dim float split vectors, nodes are large.
- int8 quantizer for split vectors helps significantly: 990MB vs 1605MB at 100k, and query
  time drops from 55ms to 12ms. The compressed split vectors mean smaller nodes = faster
  B-tree lookups = less I/O.
- Recall is underwhelming: 0.884 at 100k with 50 trees. In-memory Annoy implementations
  typically achieve >0.95 with the same tree count. The SQLite B-tree overhead adds latency
  but shouldn't affect recall — this suggests the tree construction or search algorithm may
  have quality issues.
- Profiling confirms 71% of time in pread (disk I/O). Each tree traversal does log(N) node
  reads, and with 50 trees that's 50*log(100k) ~ 850 random reads per query.

**Lesson**: Annoy's per-node-row storage model is fundamentally wrong for SQLite. Each node
lookup requires a full B-tree traversal. The fix would be to pack entire trees into single
BLOBs (like FAISS does), or at minimum pack all nodes of a tree contiguously. The int8
quantizer is a band-aid that helps 4x but doesn't fix the architecture.

### Baseline (brute-force)

Still the right choice for <50k vectors. Query time scales linearly with N:
3.4ms at 10k, 33ms at 100k, 84ms at 250k.

int8 brute-force is underrated: 0.998 recall and 20ms at 100k (40% faster than float
brute-force) with negligible complexity. If you need >0.99 recall and your dataset
is under 100k, just use brute-force int8.

## Cross-cutting observations

### Insert throughput

| Index | Insert rate (rows/s) | Notes |
|-------|---------------------|-------|
| Annoy | 95-108k | Buffer-only, no per-row work |
| Baseline | 77-91k | Simple blob append |
| Rescore | 60-73k | Quantize + store both representations |
| IVF | 2.2-27k | Nearest-centroid lookup per insert, degrades with N |
| DiskANN | 80-250 | Full graph update per insert |

### Storage efficiency

At 100k vectors (768-dim float32, ~3MB raw vector data per 1k vectors):

| Index | DB Size (MB) | Overhead vs brute |
|-------|-------------|-------------------|
| Brute-float | 296 | 1.0x |
| IVF-n128-p16 | 311 | 1.05x |
| Rescore-bit-os8 | 403 | 1.36x |
| Rescore-int8-os4 | 467 | 1.58x |
| Annoy-t10 | 947 | 3.2x |
| Annoy-t50-int8 | 990 | 3.3x |
| Annoy-t50 | 1605 | 5.4x |
| Annoy-t100 | 2424 | 8.2x |

IVF is storage-efficient because vectors live in cell BLOBs (similar to chunks).
Annoy's per-node storage model causes massive bloat.

### The I/O wall

Profiling reveals that all index types are I/O-bound on disk:
- Brute-force: 76% pread (sequential chunk scan)
- IVF: 57% pread (cell scan within probed clusters)
- Annoy: 71% pread (random node lookups across trees)
- Rescore: only 14% pread (because the quantized scan fits more in cache)

Rescore's advantage is that the coarse pass scans compact quantized data (1 bit or 1 byte
per dimension instead of 4 bytes), so the working set fits in CPU cache and pread overhead
drops dramatically. This is why rescore has the lowest absolute query times at small scale.

### Query time scaling (ms, best config per type)

| N | Brute | Rescore | IVF | Annoy |
|---|-------|---------|-----|-------|
| 10k | 3.4 | 1.0 | 1.3 | 5.8 |
| 100k | 33.4 | 8.3 | 6.4 | 12.1 |
| 250k | 83.8 | 20.6 | 9.5 | 32.6 |

Brute and rescore scale linearly (O(N)). IVF scales sub-linearly (O(N/nlist * nprobe)).
Annoy should scale as O(n_trees * log(N)) but the constant factor (SQLite B-tree lookups)
is so large that it's worse than brute-force at 100k+ with float split vectors.

## Recommendations

1. **For <50k vectors**: Use brute-force (float or int8). It's fast enough and perfect recall.

2. **For 50k-500k vectors with writes**: Use rescore-int8-os4. Best balance of insert speed
   (70k rows/s), query speed (2-4x faster than brute), recall (0.99), and zero training cost.

3. **For 50k-500k vectors, read-heavy**: Use IVF with nlist=sqrt(N), nprobe=nlist/8.
   Best query latency (5-10x faster than brute) but requires training step.

4. **Annoy**: Not recommended in current form. Storage bloat and query scaling make it
   worse than rescore at every practical operating point. Needs architectural rework
   (pack tree nodes into BLOBs).

5. **DiskANN**: Not ready. Insert speed is 400x too slow, int8 quantizer is broken.
   Needs batched graph construction and quantizer debugging.

## Final summary

**Rescore is the clear winner for general use.** Zero training, fast inserts, good recall.
IVF wins on raw query speed at scale but the training cost makes it impractical for
write-heavy workloads. Annoy and DiskANN need significant implementation work before
they're competitive.
