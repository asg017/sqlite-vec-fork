# Annoy Benchmark Results: 50k vectors

Dataset: COHERE 768-dim cosine, 50k vectors, k=10, n=50 queries

```
                name       N                     index config   ins(s)  bld(s)      MB  qry(ms)  recall
--------------------------------------------------------------------------------------------------------------
           annoy-t10   50000 annoy  n_trees=10  search_k=auto      8.1    13.4   472.1    81.60  0.6660
           annoy-t25   50000 annoy  n_trees=25  search_k=auto      4.5    41.2   594.4   277.74  0.8600
           annoy-t50   50000 annoy  n_trees=50  search_k=auto      7.9    67.7   798.3   391.53  0.9260
          annoy-t100   50000 annoy  n_trees=100 search_k=auto      2.6   127.3  1205.4   660.75  0.9720
         brute-float   50000                    brute   float      1.9       -   148.4    90.82  1.0000
          brute-int8   50000    brute    int8  (rescore os=8)      2.7       -   185.4    59.07  0.9980
           brute-bit   50000    brute     bit  (rescore os=8)      2.0       -   153.2    21.62  0.8460
```

## Observations

- Recall is much better after cosine-aware split fix (t50=0.926, t100=0.972)
- Query time is a bottleneck: per-node SQLite reads dominate
- annoy-t50 at 391ms is 4x slower than brute-force float (91ms)
- DB size is large: 798MB for t50 vs 148MB for brute float
- Build time scales linearly: ~1.3s/tree at 50k

## Next steps

- Batch node reads (IN clause) to reduce SQLite round-trips
- Node caching for upper tree levels
- Consider lowering search_k auto multiplier with recall tuning
