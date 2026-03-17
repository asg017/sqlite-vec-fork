#!/usr/bin/env python3
"""Run KNN benchmarks for DiskANN and baseline configs, measuring latency + recall."""
import argparse
import os
import sqlite3
import statistics
import sys
import time

def load_query_vectors(base_db_path, n):
    """Load query vector ids and raw bytes from base.db."""
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n",
        {"n": n},
    ).fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_PATH = os.path.join(_SCRIPT_DIR, "..", "dist", "vec0")
BASE_DB = os.path.join(_SCRIPT_DIR, "seed", "base.db")

# Map config names to query modes
QUERY_MODES = {
    "diskann-binary": "diskann",
    "diskann-binary-R24": "diskann",
    "diskann-binary-R48": "diskann",
    "diskann-binary-L64": "diskann",
    "diskann-binary-L256": "diskann",
    "diskann-int8": "diskann",
    "baseline-float": "float",
    "baseline-bq": "bq-rescore",
    "baseline-int8": "int8-rescore",
}

OVERSAMPLE = 8


def load_ground_truth(gt_db_path, n_queries, k):
    """Load ground truth from a per-subset ground_truth.{size}.db file."""
    conn = sqlite3.connect(gt_db_path)
    truth = {}
    rows = conn.execute(
        "SELECT query_vector_id, neighbor_id FROM ground_truth "
        "WHERE query_vector_id < :n AND rank < :k "
        "ORDER BY query_vector_id, rank",
        {"n": n_queries, "k": k},
    ).fetchall()
    for qid, nid in rows:
        truth.setdefault(qid, []).append(int(nid))
    conn.close()
    return truth


def get_query_sql(mode, k):
    if mode == "diskann":
        return (
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH :query AND k = :k"
        )
    elif mode == "float":
        return (
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH :query AND k = :k"
        )
    elif mode == "bq-rescore":
        return (
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_bq MATCH vec_quantize_binary(:query)"
            "  LIMIT :oversample_k"
            ") "
            "SELECT id, vec_distance_cosine(embedding, :query) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k"
        )
    elif mode == "int8-rescore":
        return (
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_int8 MATCH vec_quantize_int8(:query, 'unit')"
            "  LIMIT :oversample_k"
            ") "
            "SELECT id, vec_distance_cosine(embedding, :query) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k"
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_benchmark(db_path, ext_path, base_db, gt_db_path, config_name, n, k):
    mode = QUERY_MODES[config_name]

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)

    query_vectors = load_query_vectors(base_db, n)
    ground_truth = load_ground_truth(gt_db_path, max(qid for qid, _ in query_vectors) + 1, k)

    sql = get_query_sql(mode, k)
    oversample_k = k * OVERSAMPLE

    times = []
    recalls = []
    total = len(query_vectors)
    wall_start = time.perf_counter()

    for i, (qid, query) in enumerate(query_vectors):
        params = {"query": query, "k": k, "oversample_k": oversample_k}
        t0 = time.perf_counter()
        results = conn.execute(sql, params).fetchall()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

        result_ids = set(row[0] for row in results)
        true_ids = set(ground_truth.get(qid, [])[:k])
        if true_ids:
            recall = len(result_ids & true_ids) / len(true_ids)
        else:
            recall = 0.0
        recalls.append(recall)

        # progress bar
        done = i + 1
        pct = done / total
        wall_elapsed = time.perf_counter() - wall_start
        avg_ms = (wall_elapsed / done) * 1000
        eta = (wall_elapsed / done) * (total - done)
        bar_w = 30
        filled = int(bar_w * pct)
        bar = "#" * filled + "." * (bar_w - filled)
        print(
            f"\r  [{bar}] {done}/{total}  "
            f"avg {avg_ms:.0f}ms  "
            f"eta {eta:.1f}s",
            end="", flush=True,
        )

    print()
    conn.close()
    return times, recalls


def summarize(times, recalls):
    times_ms = [t * 1000 for t in times]
    mean_time = statistics.mean(times_ms)
    median_time = statistics.median(times_ms)
    p99_time = sorted(times_ms)[int(len(times_ms) * 0.99)] if len(times_ms) > 1 else times_ms[0]
    total_time = sum(times_ms)
    mean_recall = statistics.mean(recalls)
    return {
        "mean_ms": round(mean_time, 3),
        "median_ms": round(median_time, 3),
        "p99_ms": round(p99_time, 3),
        "total_ms": round(total_time, 3),
        "qps": round(len(times) / (total_time / 1000), 1),
        "recall": round(mean_recall, 4),
    }


def save_result(out_dir, config_name, subset_size, k, n, stats, db_path):
    results_path = os.path.join(out_dir, "results.db")
    db = sqlite3.connect(results_path)
    schema_path = os.path.join(os.path.dirname(__file__), "SCHEMA.sql")
    with open(schema_path) as f:
        db.executescript(f.read())
    db.execute(
        "INSERT OR REPLACE INTO bench_results "
        "(config_name, subset_size, k, n, mean_ms, median_ms, p99_ms, "
        "total_ms, qps, recall, db_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (config_name, subset_size, k, n,
         stats["mean_ms"], stats["median_ms"], stats["p99_ms"],
         stats["total_ms"], stats["qps"], stats["recall"], db_path),
    )
    db.commit()
    db.close()


def main():
    all_configs = list(QUERY_MODES.keys())

    parser = argparse.ArgumentParser(description="DiskANN vs baseline KNN benchmark")
    parser.add_argument("--config", required=True, choices=all_configs, help="config to benchmark")
    parser.add_argument("--subset-size", type=int, required=True)
    parser.add_argument("-k", type=int, default=10, help="number of nearest neighbors")
    parser.add_argument("-n", type=int, default=100, help="number of query vectors")
    parser.add_argument("--base-db", default=BASE_DB)
    parser.add_argument("--ext", default=EXT_PATH)
    parser.add_argument("-o", "--out-dir", default=".", help="output directory (with results.db)")
    args = parser.parse_args()

    db_path = os.path.join(args.out_dir, f"{args.config}.{args.subset_size}.db")
    gt_path = os.path.join(args.out_dir, f"ground_truth.{args.subset_size}.db")

    if not os.path.exists(db_path):
        print(f"ERROR: index db not found: {db_path}")
        sys.exit(1)
    if not os.path.exists(gt_path):
        print(f"ERROR: ground truth not found: {gt_path}")
        sys.exit(1)

    mode = QUERY_MODES[args.config]
    print(f"Benchmarking {args.config} (mode={mode}), n={args.n}, k={args.k}")
    print(f"  db: {db_path}")
    print(f"  gt: {gt_path}")

    times, recalls = run_benchmark(
        db_path, args.ext, args.base_db, gt_path, args.config, args.n, args.k
    )
    stats = summarize(times, recalls)

    print(f"  mean:   {stats['mean_ms']:>10.3f} ms")
    print(f"  median: {stats['median_ms']:>10.3f} ms")
    print(f"  p99:    {stats['p99_ms']:>10.3f} ms")
    print(f"  qps:    {stats['qps']:>10.1f}")
    print(f"  recall: {stats['recall']:>10.4f}")

    save_result(args.out_dir, args.config, args.subset_size, args.k, args.n, stats, db_path)
    print(f"\nResults saved to {os.path.join(args.out_dir, 'results.db')}")


if __name__ == "__main__":
    main()
