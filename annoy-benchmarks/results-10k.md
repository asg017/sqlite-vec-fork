# Annoy Benchmark Results: 10k vectors

Dataset: COHERE 768-dim cosine, 10k vectors, k=10, n=50 queries

```
                name       N                     index config   ins(s)  bld(s)      MB  qry(ms)  recall
--------------------------------------------------------------------------------------------------------------
           annoy-t10   10000 annoy  n_trees=10  search_k=auto      2.7     0.8    84.5     5.44  0.0560
           annoy-t25   10000 annoy  n_trees=25  search_k=auto      0.4     1.7    93.5    10.97  0.2340
           annoy-t50   10000 annoy  n_trees=50  search_k=auto      0.4     3.3   108.5    19.11  0.4740
          annoy-t100   10000 annoy  n_trees=100 search_k=auto      0.4     6.4   138.5    28.72  1.0000
         brute-float   10000                    brute   float      0.3       -    30.3    17.19  1.0000
          brute-int8   10000    brute    int8  (rescore os=8)      0.4       -    37.9    19.99  0.9980
           brute-bit   10000    brute     bit  (rescore os=8)     15.0       -    31.3    13.01  0.8520
```

## Observations

- annoy-t100 achieves perfect recall (1.0) at 10k, validating correctness
- Lower tree counts have poor recall — the auto search_k formula or the
  two-means split implementation may need tuning for high-dim cosine data
- Insert time is fast (~25k rows/s), build time scales linearly with n_trees
- DB size is larger than baselines due to separate node + vector tables
- Query time scales roughly linearly with n_trees
