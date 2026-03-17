# Step 9: Testing Strategy

*Reference: `plans/diskann-plans/11-testing.md` for the DiskANN testing approach.*

## Overview

Testing follows the same layered approach as DiskANN: C unit tests for
algorithms, Python integration tests for SQL behavior, and recall benchmarks
for quality validation.

## C Unit Tests

### K-Means Algorithm (`test_ivf_kmeans_*`)

```c
// 1. Basic clustering: 4 well-separated 2D clusters
void test_ivf_kmeans_basic_2d(void);

// 2. K-means++ initialization produces spread-out centroids
void test_ivf_kmeans_plusplus_spread(void);

// 3. Empty cluster re-initialization
void test_ivf_kmeans_empty_cluster_recovery(void);

// 4. Convergence: identical data converges in 1 iteration
void test_ivf_kmeans_convergence_identical(void);

// 5. Single point per cluster: k=N
void test_ivf_kmeans_k_equals_n(void);

// 6. High-dimensional: 128D random data, verify inertia decreases
void test_ivf_kmeans_high_dim(void);
```

### Parser (`test_ivf_parse_*`)

```c
// 1. Parse defaults: ivf()
void test_ivf_parse_defaults(void);

// 2. Parse all options: ivf(nlist=256, nprobe=16, metric=cosine)
void test_ivf_parse_all_options(void);

// 3. Error: nprobe > nlist
void test_ivf_parse_nprobe_gt_nlist(void);

// 4. Error: unknown key
void test_ivf_parse_unknown_key(void);

// 5. nlist=0 (deferred) is valid
void test_ivf_parse_nlist_zero(void);
```

## Python Integration Tests

### Shadow Table Tests (`test_ivf_shadow_*`)

```python
def test_ivf_shadow_tables_created(db):
    """Creating IVF table creates all three shadow tables."""

def test_ivf_shadow_tables_dropped(db):
    """DROP TABLE removes shadow tables."""

def test_ivf_info_initial_state(db):
    """ivf_trained_0 starts at '0'."""
```

### Insert Tests (`test_ivf_insert_*`)

```python
def test_ivf_insert_flat_mode(db):
    """Before training, vectors go to unassigned."""

def test_ivf_insert_trained_mode(db):
    """After training, vectors go to assigned with correct centroid."""

def test_ivf_insert_preserves_rowid(db):
    """Rowid in shadow table matches rowid in virtual table."""

def test_ivf_delete(db):
    """DELETE removes from correct shadow table."""

def test_ivf_delete_after_training(db):
    """DELETE works for assigned vectors."""
```

### Centroid Management Tests (`test_ivf_centroid_*`)

```python
def test_ivf_compute_centroids_basic(db):
    """compute-centroids runs k-means and assigns all vectors."""

def test_ivf_compute_centroids_with_options(db):
    """compute-centroids:{json} respects nlist override."""

def test_ivf_compute_centroids_recompute(db):
    """Recomputing centroids re-assigns all vectors."""

def test_ivf_set_centroid_manual(db):
    """set-centroid:N imports external centroids."""

def test_ivf_set_centroid_bulk(db):
    """Bulk import via SELECT ... INSERT."""

def test_ivf_assign_vectors(db):
    """assign-vectors moves unassigned to assigned."""

def test_ivf_assign_vectors_no_centroids_error(db):
    """assign-vectors errors if no centroids exist."""

def test_ivf_clear_centroids(db):
    """clear-centroids removes centroids and moves vectors back."""
```

### KNN Query Tests (`test_ivf_knn_*`)

```python
def test_ivf_knn_flat_mode(db):
    """Flat mode returns correct brute-force results."""

def test_ivf_knn_after_training(db):
    """IVF mode returns correct approximate results."""

def test_ivf_knn_hybrid_mode(db):
    """Hybrid mode finds vectors in both assigned and unassigned."""

def test_ivf_knn_empty_table(db):
    """KNN on empty table returns no results."""

def test_ivf_knn_k_larger_than_n(db):
    """KNN with k > number of vectors returns all vectors."""

def test_ivf_knn_exact_match(db):
    """Querying with an existing vector returns distance=0."""
```

### Edge Case Tests

```python
def test_ivf_nlist_zero_deferred(db):
    """nlist=0 at creation, nlist set at compute-centroids time."""

def test_ivf_few_vectors_many_centroids(db):
    """compute-centroids with N < nlist reduces nlist to N."""

def test_ivf_concurrent_insert_query(db):
    """Inserts during query don't crash (SQLite serialization)."""

def test_ivf_large_dimensions(db):
    """float[2048] vectors work correctly."""

def test_ivf_multiple_vector_columns(db):
    """Two vector columns, one IVF one flat, work independently."""

def test_ivf_with_auxiliary_columns(db):
    """Auxiliary (+) columns work alongside IVF vectors."""

def test_ivf_with_partition_columns(db):
    """Partition key columns work with IVF index."""
```

## Recall Benchmarks

Recall tests verify that IVF achieves acceptable quality:

```python
@pytest.mark.parametrize("nlist,nprobe,expected_recall", [
    (16, 1, 0.3),    # low nprobe, low recall OK
    (16, 4, 0.7),    # moderate nprobe
    (16, 16, 1.0),   # nprobe=nlist = brute force, perfect recall
    (64, 4, 0.4),
    (64, 16, 0.8),
    (64, 64, 1.0),
])
def test_ivf_recall(db, nlist, nprobe, expected_recall):
    """IVF recall meets minimum thresholds."""
    N = 1000
    D = 32
    # Insert random vectors
    # Query with known ground truth (brute force)
    # Compare IVF results to ground truth
    # Assert recall >= expected_recall
```

## Performance Benchmarks

Not part of the test suite, but useful for development:

```python
def bench_ivf_insert_throughput():
    """Measure inserts/second (flat mode and trained mode)."""

def bench_ivf_training_time():
    """Measure compute-centroids wall time vs N and nlist."""

def bench_ivf_query_latency():
    """Measure KNN query time vs nprobe and N."""

def bench_ivf_vs_flat():
    """Compare IVF query time to brute-force flat scan."""
```

## Files

- `sqlite-vec-ivf-kmeans.c`: K-means C unit tests (compiled with
  `#ifdef SQLITE_VEC_UNIT_TESTS`). Tests live next to the code they test.
- `sqlite-vec-ivf.c`: Parser C unit tests (same `#ifdef` guard).
- `tests/test-ivf.py`: Python integration tests.
- `tests/test-ivf-recall.py`: Recall benchmarks (separate file, slower).
