#!/usr/bin/env python3
"""Sweep nlist/nprobe at 1M vectors to find optimal IVF parameters.

Strategy: insert sqrt(N) vectors, train k-means, insert the rest.
Runs multiple configs sequentially, outputs comparison table.
"""
import os
import math
import sqlite3
import statistics
import time
import sys

EXT_PATH = os.path.join(os.path.dirname(__file__), "..", "dist", "vec0")
BASE_DB = os.path.join(
    os.path.dirname(__file__), "..", "benchmark2", "zilliz", "seed", "base.db"
)
INSERT_BATCH_SIZE = 1000
N = 1_000_000
K = 10
N_QUERIES = 50


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def run_config(name, nlist, nprobe, train_size, out_dir):
    print(f"\n{'='*70}", flush=True)
    print(f"  {name}: N={N}, nlist={nlist}, nprobe={nprobe}, train={train_size}", flush=True)
    print(f"{'='*70}", flush=True)

    db_path = os.path.join(out_dir, f"{name}.{N}.db")
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

    # Phase 1: Insert training vectors
    print(f"  Phase 1: Insert {train_size} training vectors...", flush=True)
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
        if done % 50000 == 0 or done == train_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{train_size}  {elapsed:.1f}s  {rate:.0f} rows/s", flush=True)
    phase1_time = time.perf_counter() - t0

    # Phase 2: Train k-means
    print(f"  Phase 2: Train k-means (nlist={nlist} on {train_size} vectors)...", flush=True)
    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train
    print(f"    Done in {train_time:.1f}s", flush=True)

    # Phase 3: Insert remaining (auto-assigned)
    remaining = N - train_size
    print(f"  Phase 3: Insert remaining {remaining} vectors...", flush=True)
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
            eta = (remaining - done) / rate if rate > 0 else 0
            print(f"    {done:>8}/{remaining}  {elapsed:.0f}s  {rate:.0f} rows/s  eta {eta:.0f}s", flush=True)
    phase3_time = time.perf_counter() - t_rest

    total_insert = phase1_time + phase3_time
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    print(f"  Build: {total_insert:.1f}s insert + {train_time:.1f}s train  "
          f"({file_size_mb:.0f} MB, {row_count} rows)", flush=True)

    # Measure KNN
    print(f"  Measuring KNN (k={K}, n={N_QUERIES})...", flush=True)
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

        # Ground truth from the dataset's neighbor table (pre-computed)
        # Fall back to brute-force on a subset if needed
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

    print(f"  KNN: mean={mean_ms:.1f}ms  median={median_ms:.1f}ms  recall@{K}={recall:.4f}", flush=True)

    # Clean up the large DB file
    os.remove(db_path)

    return {
        "name": name,
        "nlist": nlist,
        "nprobe": nprobe,
        "train_size": train_size,
        "insert_s": round(total_insert, 1),
        "train_s": round(train_time, 1),
        "file_mb": round(file_size_mb, 0),
        "qry_mean_ms": round(mean_ms, 1),
        "qry_median_ms": round(median_ms, 1),
        "recall": round(recall, 4),
    }


def run_baseline_flat(out_dir):
    """vec0 brute-force flat baseline at 1M."""
    print(f"\n{'='*70}", flush=True)
    print(f"  vec0-flat: N={N} (brute-force baseline)", flush=True)
    print(f"{'='*70}", flush=True)

    db_path = os.path.join(out_dir, f"vec0-flat.{N}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

    conn.execute(
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key,"
        "  embedding float[768] distance_metric=cosine"
        ")"
    )

    t0 = time.perf_counter()
    for lo in range(0, N, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, N)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
        done = hi
        if done % 100000 == 0:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{N}  {elapsed:.0f}s  {rate:.0f} rows/s", flush=True)
    insert_time = time.perf_counter() - t0
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    print(f"  Build: {insert_time:.1f}s  ({file_size_mb:.0f} MB)", flush=True)

    # Measure KNN (only 10 queries — brute-force at 1M is slow)
    n_queries = 10
    print(f"  Measuring KNN (k={K}, n={n_queries})...", flush=True)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

    queries = load_query_vectors(BASE_DB, n_queries)
    times_ms = []
    for qid, query in queries:
        t0 = time.perf_counter()
        conn.execute(
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH :query AND k = :k",
            {"query": query, "k": K},
        ).fetchall()
        times_ms.append((time.perf_counter() - t0) * 1000)

    conn.close()
    mean_ms = statistics.mean(times_ms)
    print(f"  KNN: mean={mean_ms:.0f}ms (brute-force, {n_queries} queries)", flush=True)

    os.remove(db_path)

    return {
        "name": "vec0-flat",
        "nlist": 0,
        "nprobe": 0,
        "train_size": 0,
        "insert_s": round(insert_time, 1),
        "train_s": 0,
        "file_mb": round(file_size_mb, 0),
        "qry_mean_ms": round(mean_ms, 1),
        "qry_median_ms": round(statistics.median(times_ms), 1),
        "recall": 1.0,
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)

    # Configs to sweep: (name, nlist, nprobe, train_size)
    # train_size = min(N, 8*nlist) — keep k-means tractable
    configs = [
        # Small nlist, varying nprobe
        ("ivf-n64-p4",    64,   4, min(N, 8*64)),
        ("ivf-n64-p16",   64,  16, min(N, 8*64)),
        ("ivf-n64-p32",   64,  32, min(N, 8*64)),

        # Medium nlist
        ("ivf-n128-p8",  128,   8, min(N, 8*128)),
        ("ivf-n128-p16", 128,  16, min(N, 8*128)),
        ("ivf-n128-p32", 128,  32, min(N, 8*128)),

        # Larger nlist
        ("ivf-n256-p8",  256,   8, min(N, 8*256)),
        ("ivf-n256-p16", 256,  16, min(N, 8*256)),
        ("ivf-n256-p32", 256,  32, min(N, 8*256)),
        ("ivf-n256-p64", 256,  64, min(N, 8*256)),
    ]

    all_results = []

    # Baseline first
    all_results.append(run_baseline_flat(out_dir))

    for name, nlist, nprobe, train_size in configs:
        result = run_config(name, nlist, nprobe, train_size, out_dir)
        all_results.append(result)

        # Print running summary after each config
        print(f"\n--- Running Summary ---", flush=True)
        print(f"{'name':>18} {'nlist':>6} {'nprobe':>6} {'train':>6} "
              f"{'insert(s)':>10} {'train(s)':>9} {'MB':>6} "
              f"{'qry(ms)':>8} {'recall':>8}", flush=True)
        print("-" * 95, flush=True)
        for r in all_results:
            train_str = f"{r['train_s']:.0f}" if r["train_s"] > 0 else "-"
            print(f"{r['name']:>18} {r['nlist']:>6} {r['nprobe']:>6} {r['train_size']:>6} "
                  f"{r['insert_s']:>10.1f} {train_str:>9} {r['file_mb']:>6.0f} "
                  f"{r['qry_mean_ms']:>8.1f} {r['recall']:>8.4f}", flush=True)

    # Final report
    print(f"\n\n{'='*95}", flush=True)
    print(f"FINAL RESULTS — 1M vectors, COHERE 768-dim cosine", flush=True)
    print(f"{'='*95}", flush=True)
    print(f"{'name':>18} {'nlist':>6} {'nprobe':>6} {'train':>6} "
          f"{'insert(s)':>10} {'train(s)':>9} {'MB':>6} "
          f"{'qry(ms)':>8} {'recall':>8}", flush=True)
    print("-" * 95, flush=True)
    for r in all_results:
        train_str = f"{r['train_s']:.0f}" if r["train_s"] > 0 else "-"
        print(f"{r['name']:>18} {r['nlist']:>6} {r['nprobe']:>6} {r['train_size']:>6} "
              f"{r['insert_s']:>10.1f} {train_str:>9} {r['file_mb']:>6.0f} "
              f"{r['qry_mean_ms']:>8.1f} {r['recall']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
