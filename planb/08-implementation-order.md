# Step 8: Implementation Order and Dependencies

## Overview

The IVF implementation is designed to be built incrementally, with each step
independently testable. This mirrors the 11-step DiskANN approach
(see `plans/diskann-plans/`) but with significantly fewer steps due to IVF's
simpler architecture.

## Step Dependency Graph

```
Step 1: Data Structures
  │
  ├── Step 2: Parse INDEXED BY ivf(...)
  │     │
  │     └── Step 3: Shadow Tables
  │           │
  │           ├── Step 4: Insert Path (flat mode only)
  │           │     │
  │           │     └── Step 7: KNN Query (flat mode only)
  │           │
  │           └── Step 5: K-Means Implementation (pure C, no SQLite)
  │                 │
  │                 └── Step 6: Centroid Management Commands
  │                       │
  │                       ├── Step 4+: Insert Path (trained mode)
  │                       │
  │                       └── Step 7+: KNN Query (IVF probe mode)
```

## Recommended Build Order

### Phase 1: Foundation (Steps 1-3)
Minimal plumbing. After this phase, `CREATE VIRTUAL TABLE ... USING vec0(...
indexed by ivf(...))` works and creates shadow tables.

### Phase 2: Flat Mode (Steps 4 + 7, flat mode only)
Insert vectors into unassigned table, brute-force KNN query. The table is
fully functional (but slow) — equivalent to the existing vec0 flat index.
**This is a useful checkpoint** — you can ship this and it works, just without
the IVF speedup.

### Phase 3: K-Means (Step 5)
Pure C implementation, no SQLite integration. Can be unit-tested independently
with synthetic data. **This is the most parallelizable step** — it can be
developed concurrently with Phase 2.

### Phase 4: Training (Step 6)
Wire k-means into the command interface. After this, `compute-centroids` and
`set-centroid` work. Vectors get assigned to cells.

### Phase 5: IVF Query (Step 7, IVF probe mode)
The final step — KNN queries probe cells instead of scanning everything.

## File Layout

All IVF code goes into two new files, `#include`d into `sqlite-vec.c`:

```
sqlite-vec-ivf-kmeans.c    Steps 5        (~250 lines, zero SQLite deps)
sqlite-vec-ivf.c           Steps 1-4,6,7  (~850 lines, all IVF SQLite logic)
sqlite-vec.c               Dispatch only   (~50 lines of additions)
```

The `#include`s go in `sqlite-vec.c` after the struct definitions that the IVF
code depends on (`vec0_vtab`, `VectorColumnDefinition`, etc.):

```c
// In sqlite-vec.c, after struct definitions:
#include "sqlite-vec-ivf-kmeans.c"
#include "sqlite-vec-ivf.c"
```

No Makefile changes. The preprocessor inlines everything into one translation
unit.

## Estimated Complexity per Step

| Step | Description | New file lines (est.) | sqlite-vec.c lines added |
|------|-------------|----------------------|--------------------------|
| 1 | Data structures | ~50 in `sqlite-vec-ivf.c` | ~10 (struct fields) |
| 2 | Parser | ~80 in `sqlite-vec-ivf.c` | ~3 (dispatch) |
| 3 | Shadow tables | ~100 in `sqlite-vec-ivf.c` | ~5 (create/destroy calls) |
| 4 | Insert path | ~120 in `sqlite-vec-ivf.c` | ~8 (insert/delete hooks) |
| 5 | K-means | ~250 in `sqlite-vec-ivf-kmeans.c` | ~1 (`#include`) |
| 6 | Centroid commands | ~300 in `sqlite-vec-ivf.c` | ~8 (command dispatch) |
| 7 | KNN query | ~200 in `sqlite-vec-ivf.c` | ~6 (filter/bestindex) |
| **Total** | | **~1100 in new files** | **~41 in sqlite-vec.c** |

Compare to DiskANN: ~2500+ lines added directly to `sqlite-vec.c` across 11
steps.

## What's NOT Needed (vs. DiskANN)

- No graph data structure or blob encoding (DiskANN steps 4, 5)
- No RobustPrune algorithm (DiskANN step 7)
- No graph repair on delete (DiskANN step 9)
- No medoid management (DiskANN step 5)
- No batched insert buffer (DiskANN step 11) — IVF inserts are cheap
- No quantized distance functions — centroids use full-precision vectors

## Testing Strategy

Each step should include:
- **C unit tests** for pure functions (k-means, parser).
- **Python integration tests** for SQL-level behavior.
- **Recall tests** (Step 7) comparing IVF results to brute-force ground truth.

See `plans/diskann-plans/11-testing.md` for the testing framework pattern.

## Future Enhancements (Not in Initial Implementation)

These are explicitly out of scope for v1 but worth noting:

1. **Product quantization (PQ)** — compress stored vectors for less I/O per cell.
2. **Residual vectors** — store vector - centroid for better PQ compression.
3. **Multi-probe LSH** — probe nearby cells based on query-centroid geometry.
4. **nprobe override at query time** — hidden column for per-query tuning.
5. **Mini-batch k-means** — for datasets too large to fit in memory.
6. **Incremental centroid updates** — update centroids without full retraining.
7. **IVF-PQ** — combine IVF with product quantization for billion-scale.
