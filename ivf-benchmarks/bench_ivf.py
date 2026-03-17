#!/usr/bin/env python3
"""Benchmark IVF index vs vec0 baselines on COHERE 768-dim cosine dataset.

Configs:
  - vec0-flat:      brute-force float (baseline)
  - vec0-int8:      int8 quantized + rescore
  - vec0-bit:       binary quantized + rescore
  - ivf-flat:       IVF index, no k-means (brute-force via unassigned table)
  - ivf-trained:    IVF index, k-means trained on random sample

Usage:
  python3 bench_ivf.py                    # defaults: 10k vectors, all configs
  python3 bench_ivf.py --subset-size 5000 # smaller subset
  python3 bench_ivf.py -o runs/test1      # custom output dir
"""
import argparse
import os
import random
import sqlite3
import statistics
import struct
import sys
import time

EXT_PATH = os.path.join(os.path.dirname(__file__), "..", "dist", "vec0")
BASE_DB = os.path.join(
    os.path.dirname(__file__), "..", "benchmark2", "zilliz", "seed", "base.db"
)
INSERT_BATCH_SIZE = 1000


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


# ============================================================================
# Build functions
# ============================================================================


def insert_loop(conn, sql, subset_size, label=""):
    t0 = time.perf_counter()
    for lo in range(0, subset_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, subset_size)
        conn.execute(sql, {"lo": lo, "hi": hi})
        conn.commit()
        done = hi
        if done % 2000 == 0 or done == subset_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(
                f"    [{label}] {done:>8}/{subset_size}  "
                f"{elapsed:.1f}s  {rate:.0f} rows/s",
                flush=True,
            )
    return time.perf_counter() - t0


def build_vec0_flat(conn, subset_size):
    conn.execute(
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key,"
        "  embedding float[768] distance_metric=cosine"
        ")"
    )
    elapsed = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        subset_size,
        "flat",
    )
    return elapsed


def build_vec0_int8(conn, subset_size):
    conn.execute(
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key,"
        "  embedding float[768] distance_metric=cosine,"
        "  embedding_int8 int8[768]"
        ")"
    )
    elapsed = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding, embedding_int8) "
        "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        subset_size,
        "int8",
    )
    return elapsed


def build_vec0_bit(conn, subset_size):
    conn.execute(
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key,"
        "  embedding float[768] distance_metric=cosine,"
        "  embedding_bq bit[768]"
        ")"
    )
    elapsed = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding, embedding_bq) "
        "SELECT id, vector, vec_quantize_binary(vector) "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        subset_size,
        "bit",
    )
    return elapsed


def build_ivf_flat(conn, subset_size, nlist):
    """IVF index with NO training — stays in flat/unassigned mode."""
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    indexed by ivf(nlist={nlist})"
        f")"
    )
    elapsed = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        subset_size,
        "ivf-flat",
    )
    return elapsed


def build_ivf_trained(conn, base_db, subset_size, nlist, nprobe, train_sample):
    """IVF index with k-means training from a random sample."""
    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    indexed by ivf(nlist={nlist}, nprobe={nprobe})"
        f")"
    )

    # Insert all vectors (goes to unassigned since not yet trained)
    insert_elapsed = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        subset_size,
        "ivf-insert",
    )

    # Train k-means
    print(f"    Training k-means (nlist={nlist}, sample={train_sample})...", flush=True)
    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_elapsed = time.perf_counter() - t_train
    print(f"    Training done in {train_elapsed:.2f}s", flush=True)

    return insert_elapsed, train_elapsed


# ============================================================================
# Query functions
# ============================================================================


def query_match(conn, query, k):
    """Standard MATCH query (vec0 flat or IVF)."""
    return conn.execute(
        "SELECT id, distance FROM vec_items "
        "WHERE embedding MATCH :query AND k = :k",
        {"query": query, "k": k},
    ).fetchall()


def query_int8_rescore(conn, query, k, oversample=8):
    return conn.execute(
        "WITH coarse AS ("
        "  SELECT id, embedding FROM vec_items"
        "  WHERE embedding_int8 MATCH vec_quantize_int8(:query, 'unit')"
        "  LIMIT :oversample_k"
        ") "
        "SELECT id, vec_distance_cosine(embedding, :query) as distance "
        "FROM coarse ORDER BY 2 LIMIT :k",
        {"query": query, "k": k, "oversample_k": k * oversample},
    ).fetchall()


def query_bit_rescore(conn, query, k, oversample=8):
    return conn.execute(
        "WITH coarse AS ("
        "  SELECT id, embedding FROM vec_items"
        "  WHERE embedding_bq MATCH vec_quantize_binary(:query)"
        "  LIMIT :oversample_k"
        ") "
        "SELECT id, vec_distance_cosine(embedding, :query) as distance "
        "FROM coarse ORDER BY 2 LIMIT :k",
        {"query": query, "k": k, "oversample_k": k * oversample},
    ).fetchall()


def measure_knn(db_path, query_fn, base_db, subset_size, k=10, n_queries=50):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    queries = load_query_vectors(base_db, n_queries)

    times_ms = []
    recalls = []
    for qid, query in queries:
        t0 = time.perf_counter()
        results = query_fn(conn, query, k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        result_ids = set(r[0] for r in results)

        # Ground truth: brute-force over the subset
        gt_rows = conn.execute(
            "SELECT id FROM ("
            "  SELECT id, vec_distance_cosine(vector, :query) as dist "
            "  FROM base.train WHERE id < :n ORDER BY dist LIMIT :k"
            ")",
            {"query": query, "k": k, "n": subset_size},
        ).fetchall()
        gt_ids = set(r[0] for r in gt_rows)

        if gt_ids:
            recalls.append(len(result_ids & gt_ids) / len(gt_ids))
        else:
            recalls.append(0.0)

    conn.close()

    return {
        "mean_ms": round(statistics.mean(times_ms), 2),
        "median_ms": round(statistics.median(times_ms), 2),
        "p99_ms": round(
            sorted(times_ms)[int(len(times_ms) * 0.99)], 2
        )
        if len(times_ms) > 1
        else round(times_ms[0], 2),
        "recall": round(statistics.mean(recalls), 4),
    }


# ============================================================================
# Main
# ============================================================================


def run_config(name, build_fn, query_fn, base_db, subset_size, out_dir, k, n_queries):
    db_path = os.path.join(out_dir, f"{name}.{subset_size}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    build_result = build_fn(conn)
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()

    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    # Handle tuple return (insert_time, train_time) from ivf-trained
    if isinstance(build_result, tuple):
        insert_time, train_time = build_result
    else:
        insert_time = build_result
        train_time = 0.0

    print(
        f"  Build: {insert_time:.2f}s insert"
        + (f" + {train_time:.2f}s train" if train_time > 0 else "")
        + f"  ({file_size_mb:.1f} MB, {row_count} rows)"
    )

    print(f"  Measuring KNN (k={k}, n={n_queries})...", flush=True)
    knn = measure_knn(db_path, query_fn, base_db, subset_size, k, n_queries)
    print(
        f"  KNN: mean={knn['mean_ms']:.2f}ms  "
        f"median={knn['median_ms']:.2f}ms  "
        f"recall@{k}={knn['recall']:.4f}"
    )

    return {
        "name": name,
        "n_vectors": subset_size,
        "insert_s": round(insert_time, 3),
        "train_s": round(train_time, 3),
        "insert_per_vec_ms": round((insert_time / row_count) * 1000, 2)
        if row_count
        else 0,
        "file_size_mb": round(file_size_mb, 2),
        "knn_mean_ms": knn["mean_ms"],
        "knn_median_ms": knn["median_ms"],
        "knn_p99_ms": knn["p99_ms"],
        "recall": knn["recall"],
        "k": k,
        "n_queries": n_queries,
    }


def print_report(results):
    print(f"\n{'name':>18} {'N':>7} {'insert(s)':>10} {'train(s)':>9} "
          f"{'ms/vec':>8} {'MB':>8} {'qry(ms)':>8} {'recall':>8}")
    print("-" * 95)
    for r in results:
        train_str = f"{r['train_s']:.1f}" if r["train_s"] > 0 else "-"
        print(
            f"{r['name']:>18} {r['n_vectors']:>7} {r['insert_s']:>10.2f} "
            f"{train_str:>9} {r['insert_per_vec_ms']:>8.2f} "
            f"{r['file_size_mb']:>8.1f} {r['knn_mean_ms']:>8.2f} "
            f"{r['recall']:>8.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark IVF vs vec0 baselines")
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("-n", "--n-queries", type=int, default=50)
    parser.add_argument("--base-db", default=BASE_DB)
    parser.add_argument("--nlist", type=int, default=32,
                        help="IVF nlist (number of centroids)")
    parser.add_argument("--nprobe", type=int, default=8,
                        help="IVF nprobe (cells to probe at query time)")
    parser.add_argument("--train-sample", type=int, default=500,
                        help="Number of vectors to sample for k-means training "
                             "(0 = use all vectors)")
    parser.add_argument("-o", "--out-dir", default="runs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    N = args.subset_size

    configs = [
        (
            "vec0-flat",
            lambda conn: build_vec0_flat(conn, N),
            lambda conn, q, k: query_match(conn, q, k),
        ),
        (
            "vec0-int8",
            lambda conn: build_vec0_int8(conn, N),
            lambda conn, q, k: query_int8_rescore(conn, q, k),
        ),
        (
            "vec0-bit",
            lambda conn: build_vec0_bit(conn, N),
            lambda conn, q, k: query_bit_rescore(conn, q, k),
        ),
        (
            "ivf-flat",
            lambda conn: build_ivf_flat(conn, N, nlist=args.nlist),
            lambda conn, q, k: query_match(conn, q, k),
        ),
        (
            f"ivf-trained-n{args.nlist}-p{args.nprobe}",
            lambda conn: build_ivf_trained(
                conn, args.base_db, N,
                nlist=args.nlist, nprobe=args.nprobe,
                train_sample=args.train_sample,
            ),
            lambda conn, q, k: query_match(conn, q, k),
        ),
    ]

    all_results = []
    for i, (name, build_fn, query_fn) in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {name}")
        result = run_config(
            name, build_fn, query_fn,
            args.base_db, N, args.out_dir,
            args.k, args.n_queries,
        )
        all_results.append(result)

    print_report(all_results)


if __name__ == "__main__":
    main()
