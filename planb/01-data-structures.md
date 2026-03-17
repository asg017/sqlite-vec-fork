# Step 1: IVF Data Structures and Constants

*Reference: `plans/diskann-plans/01-data-structures.md` for the DiskANN equivalent.*

## Overview

Define C structs, enums, and constants for IVF index configuration. These are
stored per vector column in `VectorColumnDefinition`.

## Struct: `Vec0IvfConfig`

No new metric enum is needed. The distance metric is already stored in
`VectorColumnDefinition.distance_metric` (type `enum Vec0DistanceMetrics`)
and supports `L2`, `cosine`, and `ip`. IVF reuses this for centroid distance
computations and KNN queries.

```c
struct Vec0IvfConfig {
  // Whether IVF indexing is enabled for this column.
  int enabled;

  // Number of Voronoi cells / centroids.
  // Default: 128. Can be 0 at creation time (deferred to compute-centroids).
  int nlist;

  // Number of cells to probe at query time.
  // Default: 10. Must be <= nlist.
  int nprobe;
};
```

## Integration into `VectorColumnDefinition`

```c
struct VectorColumnDefinition {
  // ... existing fields ...
  struct Vec0DiskannConfig diskann;
  struct Vec0IvfConfig ivf;  // NEW
};
```

## Integration into `vec0_vtab`

```c
struct vec0_vtab {
  // ... existing fields ...

  // Per vector column: shadow table names
  char *shadowIvfCentroidsNames[VEC0_MAX_VECTOR_COLUMNS];
  char *shadowIvfVectorsNames[VEC0_MAX_VECTOR_COLUMNS];
  char *shadowIvfUnassignedNames[VEC0_MAX_VECTOR_COLUMNS];

  // Per vector column: prepared statements
  sqlite3_stmt *stmtIvfCentroidRead[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtIvfCentroidInsert[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtIvfVectorsInsert[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtIvfVectorsReadByCell[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtIvfUnassignedInsert[VEC0_MAX_VECTOR_COLUMNS];
  sqlite3_stmt *stmtIvfUnassignedReadAll[VEC0_MAX_VECTOR_COLUMNS];
};
```

## Constants

```c
#define VEC0_IVF_DEFAULT_NLIST      128
#define VEC0_IVF_DEFAULT_NPROBE      10
#define VEC0_IVF_MAX_NLIST        65536
#define VEC0_IVF_KMEANS_MAX_ITER     25
#define VEC0_IVF_KMEANS_DEFAULT_SEED  0
```

## Defaults and Validation Rules

- `nlist` must be >= 0 and <= 65536. 0 means "decide at training time."
- `nprobe` must be >= 1 and <= nlist (validated at query time if nlist=0 at
  creation).
- Distance metric comes from the column's `distance_metric=` option (existing
  vec0 feature), not from IVF config.
- Only one of `diskann` or `ivf` can be enabled per vector column.

## Files Changed

- `sqlite-vec-ivf.c` (**new**): Define `Vec0IvfConfig` struct and constants.
- `sqlite-vec.c`: Add `struct Vec0IvfConfig ivf;` field to
  `VectorColumnDefinition` and IVF prepared-statement / shadow-table-name
  fields to `vec0_vtab`. Add `#include "sqlite-vec-ivf.c"` (after structs).
  These are the only `sqlite-vec.c` struct changes for the entire IVF feature.
