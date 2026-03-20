# sqlite-vec ANN Index Benchmark Results

Benchmarked on Apple M1, macOS, using COHERE-medium-768d dataset (cosine distance).
All queries: k=10, n=100 queries. Date: 2026-03-20.

## Results at 10k vectors

| Config | Query (ms) | Recall@10 | Insert (s) | Train (s) | DB Size (MB) |
|--------|-----------|-----------|------------|-----------|--------------|
| brute-float | 3.35 | **1.000** | 0.1 | - | 30 |
| brute-int8 | 4.48 | 0.997 | 0.1 | - | 38 |
| brute-bit | 3.70 | 0.862 | 0.1 | - | 31 |
| rescore-bit-os8 | **0.97** | 0.862 | 0.1 | - | 40 |
| rescore-bit-os16 | 1.84 | 0.944 | 0.1 | - | 40 |
| rescore-int8-os4 | 1.67 | 0.992 | 0.1 | - | 47 |
| ivf-n32-p8 | 1.34 | 0.947 | 0.4 | 2.7 | 33 |
| ivf-n64-p16 | 1.48 | 0.967 | 0.4 | 5.2 | 37 |
| ivf-n128-p32 | 1.87 | 0.984 | 0.4 | 10.3 | 46 |
| diskann-R48-bin | 3.75 | 0.919 | 123.9 | - | 118 |
| diskann-R72-bin | 5.03 | 0.915 | 130.7 | - | 118 |
| diskann-R72-int8 | 2.09 | **0.009** | 42.7 | - | 586 |
| annoy-t10 | 2.42 | 0.641 | 0.1 | 1.5 | 94 |
| annoy-t50 | 6.24 | 0.966 | 0.1 | 7.4 | 158 |
| annoy-t100 | 9.85 | 0.988 | 0.1 | 14.7 | 238 |
| annoy-t50-int8 | 5.79 | 0.972 | 0.1 | 7.3 | 99 |
| annoy-t50-bin | 5.78 | 0.887 | 0.1 | 7.3 | 85 |

## Results at 100k vectors

| Config | Query (ms) | Recall@10 | Insert (s) | Train (s) | DB Size (MB) |
|--------|-----------|-----------|------------|-----------|--------------|
| brute-float | 33.38 | **1.000** | 1.2 | - | 296 |
| brute-int8 | 19.67 | 0.998 | 1.3 | - | 370 |
| rescore-bit-os8 | **8.26** | 0.865 | 1.4 | - | 403 |
| rescore-bit-os16 | 15.88 | 0.934 | 1.4 | - | 403 |
| rescore-int8-os4 | 16.17 | 0.990 | 1.4 | - | 467 |
| rescore-int8-os8 | 19.36 | 0.998 | 1.4 | - | 467 |
| ivf-n64-p16 | 11.75 | 0.987 | 19.2 | 52.6 | 305 |
| ivf-n128-p16 | **6.39** | 0.957 | 19.6 | 104.6 | 311 |
| ivf-n256-p32 | 7.16 | 0.976 | 19.9 | 207.9 | 326 |
| annoy-t10 | 4.80 | 0.502 | 1.0 | 17.1 | 947 |
| annoy-t50 | 55.06 | 0.884 | 1.0 | 85.3 | 1605 |
| annoy-t100 | 120.00 | 0.939 | 1.1 | 168.5 | 2424 |
| annoy-t50-int8 | 12.07 | 0.884 | 1.1 | 84.5 | 990 |

## Results at 250k vectors

| Config | Query (ms) | Recall@10 | Insert (s) | Train (s) | DB Size (MB) |
|--------|-----------|-----------|------------|-----------|--------------|
| brute-float | 83.77 | **1.000** | 2.8 | - | 740 |
| rescore-bit-os8 | **20.56** | 0.881 | 3.5 | - | 1008 |
| rescore-int8-os4 | 43.83 | 0.993 | 4.2 | - | 1168 |
| rescore-int8-os8 | 49.85 | 0.997 | 3.5 | - | 1168 |
| ivf-n128-p16 | **14.87** | 0.964 | 112.8 | 263.0 | 759 |
| ivf-n256-p32 | 16.00 | 0.975 | 114.9 | 528.8 | 769 |
| ivf-n512-p32 | **9.47** | 0.964 | 117.8 | 1070.5 | 796 |
| annoy-t50-int8 | 32.63 | 0.875 | 2.3 | 231.0 | 2479 |

## CPU Profiling (10k, macOS `sample`)

| Function | brute-float | rescore-bit | ivf-n64 | annoy-t50 |
|----------|------------|-------------|---------|-----------|
| pread (disk I/O) | 75.8% | 13.7% | 57.3% | **70.6%** |
| distance computation | 18.3% | - | 14.6% | 5.4% |
| min_idx (candidate sort) | 2.1% | **63.1%** | - | - |
| qsort | - | - | 12.0% | 7.6% |
| memmove | 0.6% | 0.9% | 3.1% | 6.3% |
| SQLite VM | 0.3% | 2.0% | 0.9% | 2.3% |
