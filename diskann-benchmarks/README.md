# DiskANN Benchmarks

Benchmarks comparing DiskANN graph-based ANN indexing against brute-force baselines (float, int8, binary quantization) on the [Zilliz COHERE 1M 768-dim cosine dataset](https://assets.zilliz.com/benchmark/cohere_medium_1m/).

## Setup

### 1. Download seed data and build base.db

```bash
cd seed/
make download   # Downloads ~3.1GB train.parquet + test/neighbors parquets
make base.db    # Converts parquets → base.db (~4.1GB SQLite)
cd ..
```

The parquet files come from:
- `https://assets.zilliz.com/benchmark/cohere_medium_1m/train.parquet` — 1M 768-dim float32 vectors
- `https://assets.zilliz.com/benchmark/cohere_medium_1m/test.parquet` — query vectors
- `https://assets.zilliz.com/benchmark/cohere_medium_1m/neighbors.parquet` — ground truth neighbors

The `base.db` schema:
```
train(id INTEGER PRIMARY KEY, vector BLOB)            -- 1M × 768-dim f32 blobs
query_vectors(id INTEGER PRIMARY KEY, vector BLOB)    -- query vectors
neighbors(query_vector_id, rank, neighbors_id TEXT)   -- ground truth
```

### 2. Build the sqlite-vec extension

```bash
cd ..          # repo root
make loadable  # builds dist/vec0.dylib
```

## Benchmark Tool: `bench_batched_insert.py`

The main benchmark script. Builds indexes, measures insert time, DB size, KNN query latency, and recall — then prints a comparison table.

### Config format

Each config is `name:key=val,key=val`. There are two types:

**DiskANN configs** (default if no `type=` key):
| Key | Default | Notes |
|-----|---------|-------|
| `L` | 128 | Search list size (paper uses 75) |
| `R` | 72 | Max neighbors per node (must be divisible by 8) |
| `quantizer` | binary | `binary` or `int8` |
| `buffer_threshold` | 0 | 0 = per-row insert, N = buffer then flush at N rows |

**Baseline configs** (when `type=` is set):
| Key | Values | Notes |
|-----|--------|-------|
| `type` | `float`, `int8`, `bit` | float = brute-force, int8/bit = quantized + rescore |
| `oversample` | 8 (default) | Rescore oversample ratio for int8/bit (fetches k × oversample candidates) |

### Usage

```bash
# Compare DiskANN vs all baselines at 10k vectors
python3 bench_batched_insert.py --subset-size 10000 \
    --configs \
        "diskann-L75:L=75,buffer_threshold=5000" \
        "brute-float:type=float" \
        "brute-int8:type=int8" \
        "brute-bit:type=bit" \
    -k 10 -n 50 -o runs/compare

# Compare DiskANN L values
python3 bench_batched_insert.py --subset-size 10000 \
    --configs "L75:L=75" "L128:L=128" \
    -k 10 -n 50 -o runs/compare-L

# Baseline with different oversample ratios
python3 bench_batched_insert.py --subset-size 10000 \
    --configs \
        "int8-os4:type=int8,oversample=4" \
        "int8-os8:type=int8,oversample=8" \
        "int8-os16:type=int8,oversample=16" \
    -o runs/oversample

# Skip recall measurement (faster, insert-only comparison)
python3 bench_batched_insert.py --subset-size 10000 \
    --configs "L75:L=75,buffer_threshold=5000" \
    --skip-recall -o runs/insert-only
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--subset-size` | (required) | Number of vectors to insert |
| `--configs` | (required) | Config specs (see above) |
| `-k` | 10 | KNN k |
| `-n` | 50 | Number of KNN queries for recall |
| `-o` | `.` | Output directory |
| `--skip-recall` | false | Skip KNN/recall measurement |
| `--base-db` | `../seed/base.db` | Path to seed database |
| `--ext` | `../../../dist/vec0` | Path to vec0 extension |

### Output format

The script prints a comparison table like:

```
            name       N                           index config   insert(s)   ms/vec       MB  qry(ms)   recall
------------------------------------------------------------------------------------------------------------------------
     diskann-L75   10000 diskann  binary  R=72  L=75  buf= 5000       113.1    11.31    137.1     3.05   0.8420
     brute-float   10000 brute     float                                0.1     0.01     30.3     7.60   1.0000
      brute-int8   10000 brute      int8  (rescore oversample=8)        0.1     0.01     37.9     5.16   0.9980
       brute-bit   10000 brute       bit  (rescore oversample=8)        0.1     0.01     31.3     4.59   0.8520
```

Columns:
- **name** — config name
- **N** — number of vectors
- **index config** — index type, quantizer, DiskANN params, rescore oversample
- **insert(s)** — total insert time in seconds
- **ms/vec** — milliseconds per vector insert
- **MB** — database file size
- **qry(ms)** — mean KNN query time in milliseconds
- **recall** — recall@k vs brute-force float ground truth

Results are also saved to `{out-dir}/results.db` in the `index_comparison` table.

## Other scripts

These are from the original benchmark suite and still work:

| Script | Purpose |
|--------|---------|
| `build.py` | Build DiskANN/baseline indexes (predefined configs) |
| `bench.py` | KNN latency + recall benchmark against precomputed ground truth |
| `gen_ground_truth.py` | Generate per-subset ground truth |
| `run_all.py` | Orchestrate: ground truth → build → bench |

## Key observations (10k vectors, 768-dim, cosine)

- DiskANN query time is **2–3× faster** than brute-force at 10k (3ms vs 7.6ms). The gap widens at larger N since DiskANN is O(log N) vs O(N).
- DiskANN insert is **~1000× slower** than baselines due to graph construction (search + prune + reverse edges per row).
- DiskANN recall (0.84) is comparable to binary quantization rescore (0.85) but below int8 rescore (0.998).
- The paper recommends L=75 for build (vs our default L=128) — using L=75 gives **2.2× faster inserts** with similar recall.
- `buffer_threshold` batches inserts into a flat table then flushes into the graph. Phase 1 defers but doesn't reduce total work. Real speedup needs in-memory batch construction (Phase 2, not yet implemented).
