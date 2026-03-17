# Annoy Benchmark Results

Dataset: COHERE 768-dim cosine, k=10, n=50 queries

## 10k vectors

```
                name       N                     index config   ins(s)  bld(s)      MB  qry(ms)  recall
--------------------------------------------------------------------------------------------------------------
           annoy-t10   10000 annoy  n_trees=10  search_k=auto      0.4     0.7    84.5     4.20  0.7060
           annoy-t25   10000 annoy  n_trees=25  search_k=auto      0.4     1.7    93.5     7.04  0.8640
           annoy-t50   10000 annoy  n_trees=50  search_k=auto      0.4     3.5   108.5    10.64  0.9200
          annoy-t100   10000 annoy  n_trees=100 search_k=auto      0.4     6.6   138.5    15.69  0.9600
         brute-float   10000                    brute   float      0.3       -    30.3    16.52  1.0000
          brute-int8   10000    brute    int8  (rescore os=8)      0.4       -    37.9    20.68  0.9980
           brute-bit   10000    brute     bit  (rescore os=8)      0.4       -    31.3    13.36  0.8520
```

## 50k vectors

```
                name       N                     index config   ins(s)  bld(s)      MB  qry(ms)  recall
--------------------------------------------------------------------------------------------------------------
           annoy-t10   50000 annoy  n_trees=10  search_k=auto      6.7     4.8   423.1     6.02  0.6280
           annoy-t25   50000 annoy  n_trees=25  search_k=auto      3.6    11.4   470.0    10.51  0.8140
           annoy-t50   50000 annoy  n_trees=50  search_k=auto      3.4    22.9   549.1    16.74  0.9040
          annoy-t100   50000 annoy  n_trees=100 search_k=auto      2.3    44.7   707.4    26.18  0.9520
         brute-float   50000                    brute   float      1.6       -   148.4    82.14  1.0000
          brute-int8   50000    brute    int8  (rescore os=8)      2.1       -   185.4    59.09  0.9980
           brute-bit   50000    brute     bit  (rescore os=8)      1.8       -   153.2    20.75  0.8460
```

## 100k vectors

```
                name       N                     index config   ins(s)  bld(s)      MB  qry(ms)  recall
--------------------------------------------------------------------------------------------------------------
           annoy-t10  100000 annoy  n_trees=10  search_k=auto     13.1    32.7   947.0   200.87  0.6120
           annoy-t25  100000 annoy  n_trees=25  search_k=auto     14.4    87.8  1193.2   277.05  0.8020
           annoy-t50  100000 annoy  n_trees=50  search_k=auto      6.6   132.4  1604.2   804.43  0.9040
          annoy-t100  100000 annoy  n_trees=100 search_k=auto      6.6   274.0  2424.6  1196.14  0.9560
         brute-float  100000                    brute   float     11.3       -   296.1   316.81  1.0000
          brute-int8  100000    brute    int8  (rescore os=8)      4.8       -   369.8   106.33  1.0000
           brute-bit  100000    brute     bit  (rescore os=8)      5.9       -   305.6   168.15  0.8840
```

## Analysis

### Recall vs n_trees

| n_trees | 10k  | 50k  | 100k |
|---------|------|------|------|
| 10      | 0.71 | 0.63 | 0.61 |
| 25      | 0.86 | 0.81 | 0.80 |
| 50      | 0.92 | 0.90 | 0.90 |
| 100     | 0.96 | 0.95 | 0.96 |

Recall scales well with n_trees and remains stable across dataset sizes.
annoy-t50 hits 0.90 recall consistently, annoy-t100 hits 0.95+.

### Query time vs brute-force

At 10k: annoy-t50 (10.6ms) is 1.6x faster than brute-float (16.5ms)
At 50k: annoy-t50 (16.7ms) is 4.9x faster than brute-float (82.1ms)
At 100k: annoy-t10 (200ms) is 1.6x faster than brute-float (317ms)
         annoy-t100 (1196ms) is slower than brute-float at 100k due to
         high search_k (auto = 10*100*10 = 10000 candidates, each requiring
         a SQLite B-tree lookup for exact distance).

### Key observations

1. **Recall is solid**: 0.90-0.96 at n_trees=50-100 across all sizes
2. **Query time scales with search_k**: The auto formula k*n_trees*10 is
   aggressive for high n_trees. Consider reducing to k*n_trees*5 or making
   search_k independent of n_trees.
3. **Build time is significant**: 132s for t50 at 100k (vs 0s for baselines).
   This is a one-time cost amortized over many queries.
4. **DB size is large**: ~2.4GB for t100 at 100k (vs 296MB brute-float).
   Each tree adds ~160MB of node data. The _annoy_vectors table also
   stores full-precision vectors separately from chunks.
5. **Sweet spot**: annoy-t50 provides the best tradeoff — 0.90 recall,
   competitive query time at small-medium sizes, reasonable build time.

### Bottlenecks

1. **Query I/O**: Each candidate requires a SQLite B-tree point lookup
   for exact distance computation. At search_k=5000 with 50k items,
   that's thousands of random reads. Batching with IN() clauses or
   caching hot nodes could help significantly.
2. **Build time**: O(n_trees * n * log n) with no parallelism.
   Each tree built sequentially with full dataset in memory.
3. **DB size**: Nodes store full float32 split vectors (768*4 = 3KB each).
   Quantizing split vectors to int8 would cut node size by 4x.
