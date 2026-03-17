#!/usr/bin/env python3
"""1M baseline benchmark: vec0-flat, vec0-int8 rescore, vec0-bit rescore."""
import os
import sqlite3
import statistics
import time

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


def insert_loop(conn, sql, label=""):
    t0 = time.perf_counter()
    for lo in range(0, N, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, N)
        conn.execute(sql, {"lo": lo, "hi": hi})
        conn.commit()
        done = hi
        if done % 100000 == 0:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    [{label}] {done:>8}/{N}  {elapsed:.0f}s  {rate:.0f} rows/s", flush=True)
    return time.perf_counter() - t0


def measure_knn(db_path, query_fn, n_queries=N_QUERIES):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

    queries = load_query_vectors(BASE_DB, n_queries)
    times_ms = []
    recalls = []
    for qid, query in queries:
        t0 = time.perf_counter()
        results = query_fn(conn, query, K)
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
        "mean_ms": round(statistics.mean(times_ms), 1),
        "median_ms": round(statistics.median(times_ms), 1),
        "recall": round(statistics.mean(recalls), 4),
    }


def run_flat(out_dir):
    print(f"\n{'='*70}\n  vec0-flat: N={N}\n{'='*70}", flush=True)
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
    insert_time = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        "flat",
    )
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Build: {insert_time:.1f}s  ({file_mb:.0f} MB, {row_count} rows)", flush=True)

    print(f"  Measuring KNN (k={K}, n={N_QUERIES})...", flush=True)
    knn = measure_knn(db_path, lambda c, q, k: c.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
        {"q": q, "k": k},
    ).fetchall())
    print(f"  KNN: mean={knn['mean_ms']}ms  recall={knn['recall']}", flush=True)
    os.remove(db_path)
    return {"name": "vec0-flat", "insert_s": round(insert_time, 1), "train_s": 0,
            "file_mb": round(file_mb, 0), **knn}


def run_int8(out_dir):
    print(f"\n{'='*70}\n  vec0-int8 (rescore): N={N}\n{'='*70}", flush=True)
    db_path = os.path.join(out_dir, f"vec0-int8.{N}.db")
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
        "  embedding float[768] distance_metric=cosine,"
        "  embedding_int8 int8[768]"
        ")"
    )
    insert_time = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding, embedding_int8) "
        "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        "int8",
    )
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Build: {insert_time:.1f}s  ({file_mb:.0f} MB, {row_count} rows)", flush=True)

    print(f"  Measuring KNN (k={K}, n={N_QUERIES})...", flush=True)
    def query_fn(conn, query, k):
        return conn.execute(
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_int8 MATCH vec_quantize_int8(:q, 'unit')"
            "  LIMIT :ok"
            ") SELECT id, vec_distance_cosine(embedding, :q) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k",
            {"q": query, "k": k, "ok": k * 8},
        ).fetchall()
    knn = measure_knn(db_path, query_fn)
    print(f"  KNN: mean={knn['mean_ms']}ms  recall={knn['recall']}", flush=True)
    os.remove(db_path)
    return {"name": "vec0-int8", "insert_s": round(insert_time, 1), "train_s": 0,
            "file_mb": round(file_mb, 0), **knn}


def run_bit(out_dir):
    print(f"\n{'='*70}\n  vec0-bit (rescore): N={N}\n{'='*70}", flush=True)
    db_path = os.path.join(out_dir, f"vec0-bit.{N}.db")
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
        "  embedding float[768] distance_metric=cosine,"
        "  embedding_bq bit[768]"
        ")"
    )
    insert_time = insert_loop(
        conn,
        "INSERT INTO vec_items(id, embedding, embedding_bq) "
        "SELECT id, vector, vec_quantize_binary(vector) "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        "bit",
    )
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Build: {insert_time:.1f}s  ({file_mb:.0f} MB, {row_count} rows)", flush=True)

    print(f"  Measuring KNN (k={K}, n={N_QUERIES})...", flush=True)
    def query_fn(conn, query, k):
        return conn.execute(
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_bq MATCH vec_quantize_binary(:q)"
            "  LIMIT :ok"
            ") SELECT id, vec_distance_cosine(embedding, :q) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k",
            {"q": query, "k": k, "ok": k * 8},
        ).fetchall()
    knn = measure_knn(db_path, query_fn)
    print(f"  KNN: mean={knn['mean_ms']}ms  recall={knn['recall']}", flush=True)
    os.remove(db_path)
    return {"name": "vec0-bit", "insert_s": round(insert_time, 1), "train_s": 0,
            "file_mb": round(file_mb, 0), **knn}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)

    results = []
    results.append(run_flat(out_dir))
    results.append(run_int8(out_dir))
    results.append(run_bit(out_dir))

    print(f"\n\n{'='*80}", flush=True)
    print(f"BASELINE RESULTS — 1M vectors, COHERE 768-dim cosine", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'name':>12} {'insert(s)':>10} {'MB':>6} {'qry(ms)':>8} {'recall':>8}", flush=True)
    print("-" * 50, flush=True)
    for r in results:
        print(f"{r['name']:>12} {r['insert_s']:>10.1f} {r['file_mb']:>6.0f} "
              f"{r['mean_ms']:>8.1f} {r['recall']:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
