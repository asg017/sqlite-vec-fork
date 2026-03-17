#!/usr/bin/env python3
"""One-off: IVF with sqrt(N) sample training vs full training.

Insert sqrt(N) vectors, train k-means, then insert the rest.
The rest auto-assign to nearest centroid (cheap O(nlist) per vector).
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
NLIST = 16
NPROBE = 4
K = 10
N_QUERIES = 50


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def build_ivf_sampled(out_dir):
    """Insert sqrt(N), train, then insert the rest."""
    sample_size = int(math.sqrt(N))
    db_path = os.path.join(out_dir, f"ivf-sampled.{N}.db")
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
        f"    indexed by ivf(nlist={NLIST}, nprobe={NPROBE})"
        f")"
    )

    # Phase 1: Insert sqrt(N) vectors (goes to unassigned)
    print(f"  Phase 1: Insert {sample_size} vectors (training sample)...")
    t0 = time.perf_counter()
    for lo in range(0, sample_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, sample_size)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
    phase1_time = time.perf_counter() - t0
    print(f"    {sample_size} vectors inserted in {phase1_time:.2f}s")

    # Phase 2: Train k-means on those sqrt(N) vectors
    print(f"  Phase 2: Train k-means (nlist={NLIST} on {sample_size} vectors)...")
    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train
    print(f"    Training done in {train_time:.2f}s")

    # Phase 3: Insert remaining vectors (auto-assigned to nearest centroid)
    remaining = N - sample_size
    print(f"  Phase 3: Insert remaining {remaining} vectors (auto-assigned)...")
    t_rest = time.perf_counter()
    for lo in range(sample_size, N, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, N)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
        done = hi - sample_size
        if done % 10000 == 0 or hi == N:
            elapsed = time.perf_counter() - t_rest
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{remaining}  {elapsed:.1f}s  {rate:.0f} rows/s", flush=True)
    phase3_time = time.perf_counter() - t_rest

    total_insert = phase1_time + phase3_time
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    print(f"  Build: {total_insert:.2f}s insert + {train_time:.2f}s train  "
          f"({file_size_mb:.1f} MB, {row_count} rows)")

    return db_path, total_insert, train_time, file_size_mb, row_count


def measure_knn(db_path):
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
    return {
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "recall": round(statistics.mean(recalls), 4),
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)

    sample_size = int(math.sqrt(N))
    print(f"=== IVF sampled training: sqrt({N})={sample_size} vectors ===")
    print(f"    nlist={NLIST}, nprobe={NPROBE}, N={N}\n")

    db_path, insert_time, train_time, file_size_mb, row_count = build_ivf_sampled(out_dir)

    print(f"\n  Measuring KNN (k={K}, n={N_QUERIES})...")
    knn = measure_knn(db_path)
    print(f"  KNN: mean={knn['mean_ms']:.2f}ms  median={knn['median_ms']:.2f}ms  "
          f"recall@{K}={knn['recall']:.4f}")

    print(f"\n{'='*70}")
    print(f"  ivf-sampled-n{NLIST}-p{NPROBE}  N={N}")
    print(f"    insert:  {insert_time:.2f}s  ({insert_time/row_count*1000:.2f} ms/vec)")
    print(f"    train:   {train_time:.2f}s  (on {sample_size} vectors)")
    print(f"    db size: {file_size_mb:.1f} MB")
    print(f"    query:   {knn['mean_ms']:.2f}ms mean, {knn['median_ms']:.2f}ms median")
    print(f"    recall:  {knn['recall']:.4f}")
    print()
    print(f"  Compare to previous runs (100k):")
    print(f"    ivf-trained-n128-p16 (full):    108.2s train,  30.87ms query, 0.8520 recall")
    print(f"    ivf-trained-n128-p16 (sampled):  0.07s train, 190.97ms query, 0.8720 recall")
    print(f"    vec0-flat:                       0s train,     71.01ms query, 1.0000 recall")
    print(f"    vec0-int8 (rescore):             0s train,     26.34ms query, 0.9980 recall")
    print(f"    vec0-bit (rescore):              0s train,     18.65ms query, 0.8840 recall")


if __name__ == "__main__":
    main()
