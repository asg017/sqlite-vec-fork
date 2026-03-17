# Annoy Benchmark Results: 100k vectors

Dataset: COHERE 768-dim cosine, 100k vectors, k=10, n=50 queries

```
                name       N                     index config   ins(s)  bld(s)      MB  qry(ms)  recall
--------------------------------------------------------------------------------------------------------------
           annoy-t10  100000 annoy  n_trees=10  search_k=auto      8.3    29.0   947.0     8.84  0.4900
           annoy-t25  100000 annoy  n_trees=25  search_k=auto      6.2    59.9  1193.2    16.42  0.7220
           annoy-t50  100000 annoy  n_trees=50  search_k=auto      5.9   121.8  1604.2    48.54  0.8940
          annoy-t100  100000 annoy  n_trees=100 search_k=auto      9.8   251.5  2424.6   746.28  0.9480
         brute-float  100000                    brute   float      4.0       -   296.1   174.92  1.0000
          brute-int8  100000    brute    int8  (rescore os=8)      4.9       -   369.8   109.42  1.0000
           brute-bit  100000    brute     bit  (rescore os=8)      4.4       -   305.6    28.89  0.8840
```

## Summary across all sizes

| Config | 10k qry | 10k recall | 50k qry | 50k recall | 100k qry | 100k recall |
|--------|---------|-----------|---------|-----------|----------|------------|
| annoy-t25 | 11.8ms | 0.872 | 15.9ms | 0.778 | 16.4ms | 0.722 |
| annoy-t50 | 14.8ms | 0.958 | 21.2ms | 0.906 | 48.5ms | 0.894 |
| annoy-t100 | 23.4ms | 0.988 | - | - | 746.3ms | 0.948 |
| brute-float | 17.9ms | 1.000 | 89.8ms | 1.000 | 174.9ms | 1.000 |
| brute-bit | 14.5ms | 0.852 | 21.3ms | 0.846 | 28.9ms | 0.884 |

## Key findings

- annoy-t50 achieves ~90% recall at all sizes, competitive with bit quant recall
- At 10k-50k, annoy-t50 is faster or equal to bit quant with better recall
- At 100k, annoy-t50 (48.5ms) is 1.7x slower than bit quant (28.9ms)
- annoy-t100 regresses at 100k due to high search_k (10000) causing many batch I/O rounds
- DB size is 5-8x larger than baselines (tree nodes + separate vector table)

## Remaining optimizations (TODOs)

- Cache upper tree levels in memory (roots + first 2-3 levels)
- PRAGMA mmap_size for memory-mapped I/O
- SIMD cosine distance (AVX not wired for cosine in sqlite-vec)
- Larger batch sizes or adaptive batching
