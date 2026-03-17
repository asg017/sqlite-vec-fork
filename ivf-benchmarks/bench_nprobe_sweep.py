#!/usr/bin/env python3
"""Build one IVF index, sweep nprobe at query time.

Usage:
  python3 bench_nprobe_sweep.py [--n 100000] [--nlist 316] [--train-mult 16]
"""
import argparse
import os
import sqlite3
import statistics
import math
import time

EXT_PATH = os.path.join(os.path.dirname(__file__), "..", "dist", "vec0")
BASE_DB = os.path.join(
    os.path.dirname(__file__), "..", "benchmark2", "zilliz", "seed", "base.db"
)
INSERT_BATCH_SIZE = 1000
K = 10
N_QUERIES = 50


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--nlist", type=int, default=0, help="0 = sqrt(N)")
    parser.add_argument("--train-mult", type=int, default=16, help="train_size = train_mult * nlist")
    parser.add_argument("--nprobes", type=str, default="1,2,4,8,16,32,64,128",
                        help="comma-separated nprobe values to test")
    args = parser.parse_args()

    N = args.n
    nlist = args.nlist if args.nlist > 0 else int(math.sqrt(N))
    train_size = min(N, args.train_mult * nlist)
    nprobes = [int(x) for x in args.nprobes.split(",")]

    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, f"nprobe_sweep.{N}.db")

    print(f"=== nprobe sweep: N={N:,}, nlist={nlist}, train={train_size} ===\n")

    # Build index once
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
        f"    indexed by ivf(nlist={nlist}, nprobe=1)"
        f")"
    )

    print(f"  Inserting {train_size} training vectors...", flush=True)
    t0 = time.perf_counter()
    for lo in range(0, train_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, train_size)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
    phase1 = time.perf_counter() - t0

    print(f"  Training k-means (nlist={nlist})...", flush=True)
    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train
    print(f"  Train: {train_time:.1f}s", flush=True)

    remaining = N - train_size
    print(f"  Inserting {remaining} remaining vectors...", flush=True)
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
        if done % 100000 == 0 or hi == N:
            elapsed = time.perf_counter() - t_rest
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{remaining}  {elapsed:.0f}s  {rate:.0f} rows/s", flush=True)
    phase3 = time.perf_counter() - t_rest
    total_insert = phase1 + phase3

    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Build done: {total_insert:.1f}s insert + {train_time:.1f}s train  ({file_mb:.0f} MB)\n", flush=True)
    conn.close()

    # Load query vectors once
    queries = load_query_vectors(BASE_DB, N_QUERIES)

    # Sweep nprobe
    results = []
    for nprobe in nprobes:
        if nprobe > nlist:
            continue

        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        conn.load_extension(EXT_PATH)
        conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

        # Set nprobe at runtime
        conn.execute(f"INSERT INTO vec_items(id) VALUES ('nprobe={nprobe}')")

        times_ms = []
        recalls = []
        for qid, query in queries:
            t0 = time.perf_counter()
            results_q = conn.execute(
                "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
                {"q": query, "k": K},
            ).fetchall()
            times_ms.append((time.perf_counter() - t0) * 1000)
            result_ids = set(r[0] for r in results_q)

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
        recall = statistics.mean(recalls)
        results.append({"nprobe": nprobe, "mean_ms": round(mean_ms, 1),
                         "recall": round(recall, 4)})
        print(f"  nprobe={nprobe:<4}  {mean_ms:>7.1f}ms  recall={recall:.4f}", flush=True)

    # Summary
    print(f"\n{'='*50}")
    print(f"N={N:,}  nlist={nlist}  train={train_size}")
    print(f"{'='*50}")
    print(f"{'nprobe':>8} {'qry(ms)':>8} {'recall':>8}")
    print("-" * 28)
    for r in results:
        print(f"{r['nprobe']:>8} {r['mean_ms']:>8.1f} {r['recall']:>8.4f}")

    os.remove(db_path)


if __name__ == "__main__":
    main()
