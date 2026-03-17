#!/usr/bin/env python3
"""Benchmark IVF with Faiss-recommended parameters.

Faiss guideline for <1M vectors: nlist = 4*sqrt(N) to 16*sqrt(N)
Training needs 30*nlist to 256*nlist vectors.

Strategy: insert training_size vectors, compute-centroids, then insert the rest.
"""
import os
import math
import sqlite3
import statistics
import time

EXT_PATH = os.path.join(os.path.dirname(__file__), "..", "dist", "vec0")
BASE_DB = os.path.join(
    os.path.dirname(__file__), "..", "benchmark2", "zilliz", "seed", "base.db"
)
INSERT_BATCH_SIZE = 1000
N = 100_000

# Faiss-recommended: 4*sqrt(N) for nlist
NLIST = int(4 * math.sqrt(N))  # 1264
# ~6% probe rate
NPROBE = max(1, NLIST // 16)   # 79
# Training: 8*nlist vectors — our C k-means is slow, keep training set small
# Faiss recommends 30*nlist minimum but we're not Faiss-fast
TRAIN_SIZE = min(N, 8 * NLIST)  # ~10k

K = 10
N_QUERIES = 50


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def run_benchmark(nlist, nprobe, train_size):
    print(f"=== IVF Faiss-recommended params ===")
    print(f"    N={N}, nlist={nlist}, nprobe={nprobe}, train_size={train_size}\n")

    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, f"ivf-faiss-n{nlist}-p{nprobe}.{N}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    indexed by ivf(nlist={nlist}, nprobe={nprobe})"
        f")"
    )

    # Phase 1: Insert training vectors (go to unassigned)
    print(f"  Phase 1: Insert {train_size} training vectors...")
    t0 = time.perf_counter()
    for lo in range(0, train_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, train_size)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
        done = hi
        if done % 10000 == 0 or done == train_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{train_size}  {elapsed:.1f}s  {rate:.0f} rows/s", flush=True)
    phase1_time = time.perf_counter() - t0

    # Phase 2: Train k-means
    print(f"\n  Phase 2: Train k-means (nlist={nlist} on {train_size} vectors)...")
    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train
    print(f"    Done in {train_time:.2f}s")

    # Phase 3: Insert remaining (auto-assigned)
    remaining = N - train_size
    if remaining > 0:
        print(f"\n  Phase 3: Insert remaining {remaining} vectors (auto-assigned)...")
        t_rest = time.perf_counter()
        for lo in range(train_size, N, INSERT_BATCH_SIZE):
            hi = min(lo + INSERT_BATCH_SIZE, N)
            conn.execute(
                "INSERT INTO vec_items(id, embedding) "
                "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
                {"lo": lo, "hi": hi},
            )
            conn.commit()
            done = hi - train_size
            if done % 10000 == 0 or hi == N:
                elapsed = time.perf_counter() - t_rest
                rate = done / elapsed if elapsed > 0 else 0
                print(f"    {done:>8}/{remaining}  {elapsed:.1f}s  {rate:.0f} rows/s", flush=True)
        phase3_time = time.perf_counter() - t_rest
    else:
        phase3_time = 0.0

    total_insert = phase1_time + phase3_time
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    print(f"\n  Build: {total_insert:.2f}s insert + {train_time:.2f}s train  "
          f"({file_size_mb:.1f} MB, {row_count} rows)")

    # Measure KNN
    print(f"\n  Measuring KNN (k={K}, n={N_QUERIES})...")
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

    queries = load_query_vectors(BASE_DB, N_QUERIES)
    times_ms = []
    recalls = []
    for qid, query in queries:
        t0 = time.perf_counter()
        results = conn.execute(
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH :query AND k = :k",
            {"query": query, "k": K},
        ).fetchall()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        result_ids = set(r[0] for r in results)

        gt_rows = conn.execute(
            "SELECT id FROM ("
            "  SELECT id, vec_distance_cosine(vector, :query) as dist "
            "  FROM base.train WHERE id < :n ORDER BY dist LIMIT :k"
            ")",
            {"query": query, "k": K, "n": N},
        ).fetchall()
        gt_ids = set(r[0] for r in gt_rows)
        if gt_ids:
            recalls.append(len(result_ids & gt_ids) / len(gt_ids))

    conn.close()

    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    recall = statistics.mean(recalls)

    print(f"  KNN: mean={mean_ms:.2f}ms  median={median_ms:.2f}ms  recall@{K}={recall:.4f}")

    print(f"\n{'='*70}")
    print(f"  ivf-faiss  N={N}  nlist={nlist}  nprobe={nprobe}  train_size={train_size}")
    print(f"    insert:  {total_insert:.2f}s  ({total_insert/row_count*1000:.2f} ms/vec)")
    print(f"    train:   {train_time:.2f}s")
    print(f"    db size: {file_size_mb:.1f} MB")
    print(f"    query:   {mean_ms:.2f}ms mean, {median_ms:.2f}ms median")
    print(f"    recall:  {recall:.4f}")
    print()
    print(f"  Previous results (100k):")
    print(f"    ivf-trained n128/p16:   10.05ms query, 0.8520 recall")
    print(f"    vec0-flat:              71.56ms query, 1.0000 recall")
    print(f"    vec0-int8 (rescore):    27.26ms query, 0.9980 recall")
    print(f"    vec0-bit (rescore):     18.54ms query, 0.8840 recall")


if __name__ == "__main__":
    run_benchmark(NLIST, NPROBE, TRAIN_SIZE)
