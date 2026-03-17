# Step 5: K-Means Implementation

## Overview

A custom k-means implementation in C, used by the `compute-centroids` command.
This is the most algorithmically complex piece of IVF, but it's a well-understood
algorithm with straightforward implementation.

## Algorithm: Lloyd's K-Means

```
Input: N vectors of dimension D, target k clusters, max_iterations, seed
Output: k centroid vectors

1. Initialize: pick k vectors as initial centroids (k-means++ or random)
2. Repeat until convergence or max_iterations:
   a. Assignment: assign each vector to nearest centroid
   b. Update: recompute each centroid as mean of assigned vectors
   c. Check convergence: if no assignments changed, stop
3. Return centroids
```

## Initialization: K-Means++

Random initialization leads to poor convergence. K-means++ gives much better
results with minimal extra cost:

```
1. Pick first centroid uniformly at random from data points.
2. For i = 2..k:
   a. For each data point x, compute D(x) = distance to nearest existing centroid.
   b. Pick next centroid with probability proportional to D(x)^2.
```

```c
static int ivf_kmeans_init_plusplus(
  const float *vectors,   // N * D floats, row-major
  int N,
  int D,
  int k,
  uint32_t seed,
  float *centroids        // output: k * D floats
) {
  // Simple PRNG (xorshift32) seeded with `seed`
  // Step 1: pick random first centroid
  // Step 2: weighted random selection for remaining k-1
  return SQLITE_OK;
}
```

## Core K-Means Loop

```c
struct IvfKmeansState {
  int N;          // number of vectors
  int D;          // dimensionality
  int k;          // number of clusters
  const float *vectors;  // N*D input data (read-only)

  float *centroids;      // k*D current centroids
  int *assignments;      // N assignments (centroid index per vector)
  float *new_centroids;  // k*D accumulator for mean computation
  int *counts;           // k counts for mean computation
};

static int ivf_kmeans(
  const float *vectors,
  int N,
  int D,
  int k,
  int max_iterations,
  uint32_t seed,
  float *out_centroids   // output: k * D floats, caller-allocated
) {
  struct IvfKmeansState state;
  // Allocate assignments, new_centroids, counts
  // Initialize centroids via k-means++

  for (int iter = 0; iter < max_iterations; iter++) {
    // === Assignment step ===
    int changed = 0;
    for (int i = 0; i < N; i++) {
      int nearest = ivf_find_nearest_centroid_mem(
        &vectors[i * D], state.centroids, D, k
      );
      if (nearest != state.assignments[i]) {
        state.assignments[i] = nearest;
        changed++;
      }
    }

    // Check convergence
    if (changed == 0) break;

    // === Update step ===
    memset(state.new_centroids, 0, k * D * sizeof(float));
    memset(state.counts, 0, k * sizeof(int));

    for (int i = 0; i < N; i++) {
      int c = state.assignments[i];
      state.counts[c]++;
      for (int d = 0; d < D; d++) {
        state.new_centroids[c * D + d] += vectors[i * D + d];
      }
    }

    for (int c = 0; c < k; c++) {
      if (state.counts[c] == 0) {
        // Empty cluster: re-initialize to a random data point
        // (or the farthest point from its nearest centroid)
        continue;
      }
      for (int d = 0; d < D; d++) {
        state.centroids[c * D + d] =
          state.new_centroids[c * D + d] / state.counts[c];
      }
    }
  }

  memcpy(out_centroids, state.centroids, k * D * sizeof(float));
  // Free temporaries
  return SQLITE_OK;
}
```

## Memory Considerations

K-means requires all vectors in memory for efficient iteration. For N vectors
of dimension D:

- Vectors: `N * D * 4` bytes (read from shadow tables)
- Centroids: `k * D * 4` bytes (small)
- Assignments: `N * 4` bytes
- Accumulators: `k * D * 4` bytes (small)

For 1M vectors at 1024 dimensions: ~4 GB for vectors alone. This is the main
scalability constraint.

**Mitigation strategies (future work):**
- Mini-batch k-means: sample a subset each iteration.
- Streaming: read vectors in batches, accumulate partial sums.
- External computation: user computes centroids outside sqlite-vec.

For the initial implementation, load all vectors into memory. Document the
memory requirement. The external centroid import path (`set-centroids`) exists
as an escape hatch for large datasets.

## Empty Cluster Handling

If a cluster becomes empty during an iteration:

1. Find the largest cluster (most assigned vectors).
2. Pick the vector farthest from that cluster's centroid.
3. Reassign that vector as the new centroid for the empty cluster.

This prevents degenerate solutions with fewer than k active clusters.

## Distance Functions

Reuse existing sqlite-vec distance functions from the column's
`distance_metric` (`enum Vec0DistanceMetrics`). For k-means clustering, L2 is
standard even if the query metric is cosine (centroids approximate the data
distribution in Euclidean space). For cosine metric, normalize vectors before
k-means.

```c
static float ivf_kmeans_distance(
  const float *a,
  const float *b,
  int D,
  enum Vec0DistanceMetrics distance_metric
) {
  // Reuse the column's existing distance_metric setting.
  // For L2 and IP: use L2 distance for clustering
  // For cosine: normalize, then use L2 (equivalent to angular distance)
  return vec_distance_l2_float(a, b, D);
}
```

## C Unit Tests

```c
// Test k-means on simple 2D data with obvious clusters
void test_kmeans_basic(void) {
  // 4 tight clusters at (0,0), (10,0), (0,10), (10,10)
  float vectors[] = {
    0.1, 0.1,  0.2, -0.1,  -0.1, 0.2,  // cluster 0
    9.9, 0.1,  10.1, -0.1,  10.0, 0.2,  // cluster 1
    0.1, 9.9,  -0.1, 10.1,  0.2, 10.0,  // cluster 2
    9.9, 9.9,  10.1, 10.1,  10.0, 10.0, // cluster 3
  };
  float centroids[4 * 2];
  ivf_kmeans(vectors, 12, 2, 4, 20, 42, centroids);
  // Verify centroids are near (0,0), (10,0), (0,10), (10,10)
  // (order may vary)
}

// Test k-means++ initialization produces spread-out centroids
void test_kmeans_plusplus_init(void) { ... }

// Test empty cluster handling
void test_kmeans_empty_cluster(void) { ... }

// Test convergence (should converge in few iterations for clean data)
void test_kmeans_convergence(void) { ... }
```

## Files Changed

- `sqlite-vec-ivf-kmeans.c` (**new**): Add `ivf_kmeans()`,
  `ivf_kmeans_init_plusplus()`, `IvfKmeansState`, helper functions. This file
  has **zero SQLite dependency** — pure C operating on float arrays. It is
  `#include`d into `sqlite-vec.c` before `sqlite-vec-ivf.c`.
- `sqlite-vec.c`: Add `#include "sqlite-vec-ivf-kmeans.c"` (one line). No
  other changes for this step.
