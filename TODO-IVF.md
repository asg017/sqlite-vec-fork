# IVF Implementation Progress

## Phase 1: Foundation
- [x] Step 1: Data structures and constants (`sqlite-vec-ivf.c`)
- [x] Step 2: Parse `INDEXED BY ivf(...)` syntax
- [x] Step 3: Shadow tables (centroids, vectors, unassigned)

## Phase 2: Flat Mode
- [x] Step 4: Insert path (flat mode — unassigned table)
- [x] Step 7a: KNN query (flat mode — brute force scan)

## Phase 3: K-Means
- [x] Step 5: K-means implementation (`sqlite-vec-ivf-kmeans.c`)

## Phase 4: Training
- [x] Step 6: Centroid management commands (compute-centroids, set-centroid, assign-vectors, clear-centroids)

## Phase 5: IVF Query
- [x] Step 7b: KNN query (IVF probe mode + hybrid mode)

## Test Coverage
- 10 C unit tests (IVF parser in test-unit.c)
- 20 Python integration tests (test-ivf.py)
- All 156 existing tests pass (no regressions)
