#!/usr/bin/env python3
"""100k tuning benchmark: baselines with oversample labels + IVF param sweep."""
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
K = 10
N_QUERIES = 50


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def insert_loop(conn, sql, total, label=""):
    t0 = time.perf_counter()
    for lo in range(0, total, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, total)
        conn.execute(sql, {"lo": lo, "hi": hi})
        conn.commit()
    elapsed = time.perf_counter() - t0
    print(f"    [{label}] {total} rows in {elapsed:.1f}s ({total/elapsed:.0f} rows/s)", flush=True)
    return elapsed


def measure_knn(db_path, query_fn):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")
    queries = load_query_vectors(BASE_DB, N_QUERIES)
    times_ms = []
    recalls = []
    for qid, query in queries:
        t0 = time.perf_counter()
        results = query_fn(conn, query, K)
        times_ms.append((time.perf_counter() - t0) * 1000)
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


def run_baseline(name, create_sql, insert_sql, query_fn, out_dir):
    print(f"\n  {name}", flush=True)
    db_path = os.path.join(out_dir, f"{name}.{N}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")
    conn.execute(create_sql)
    insert_time = insert_loop(conn, insert_sql, N, name)
    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    knn = measure_knn(db_path, query_fn)
    print(f"    {knn['mean_ms']}ms query, {knn['recall']} recall, {file_mb:.0f} MB", flush=True)
    os.remove(db_path)
    return {"name": name, "insert_s": round(insert_time, 1), "train_s": 0,
            "file_mb": round(file_mb, 0), **knn}


def run_ivf(name, nlist, nprobe, train_size, out_dir):
    print(f"\n  {name} (nlist={nlist}, nprobe={nprobe}, train={train_size})", flush=True)
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

    t0 = time.perf_counter()
    for lo in range(0, train_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, train_size)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
    phase1_time = time.perf_counter() - t0

    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train
    print(f"    train: {train_time:.1f}s on {train_size} vectors", flush=True)

    remaining = N - train_size
    t_rest = time.perf_counter()
    for lo in range(train_size, N, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, N)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
    phase3_time = time.perf_counter() - t_rest
    total_insert = phase1_time + phase3_time

    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"    insert: {total_insert:.1f}s, {file_mb:.0f} MB", flush=True)

    knn = measure_knn(db_path, lambda c, q, k: c.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
        {"q": q, "k": k}).fetchall())
    print(f"    {knn['mean_ms']}ms query, {knn['recall']} recall", flush=True)
    os.remove(db_path)
    return {"name": name, "insert_s": round(total_insert, 1),
            "train_s": round(train_time, 1), "file_mb": round(file_mb, 0), **knn}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    results = []

    nlist_sqrt = int(math.sqrt(N))  # 316

    print(f"=== 100k Tuning Benchmark (COHERE 768-dim cosine) ===\n")

    # Baselines
    results.append(run_baseline("flat",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine)",
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        lambda c, q, k: c.execute(
            "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
            {"q": q, "k": k}).fetchall(),
        out_dir))

    for oversample in [4, 8, 16]:
        results.append(run_baseline(f"int8(oversample={oversample})",
            "CREATE VIRTUAL TABLE vec_items USING vec0("
            "  id integer primary key,"
            "  embedding float[768] distance_metric=cosine,"
            "  embedding_int8 int8[768])",
            "INSERT INTO vec_items(id, embedding, embedding_int8) "
            "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
            "FROM base.train WHERE id >= :lo AND id < :hi",
            lambda c, q, k, os=oversample: c.execute(
                "WITH coarse AS ("
                "  SELECT id, embedding FROM vec_items"
                "  WHERE embedding_int8 MATCH vec_quantize_int8(:q, 'unit')"
                "  LIMIT :ok"
                ") SELECT id, vec_distance_cosine(embedding, :q) as distance "
                "FROM coarse ORDER BY 2 LIMIT :k",
                {"q": q, "k": k, "ok": k * os}).fetchall(),
            out_dir))

    for oversample in [4, 8, 16]:
        results.append(run_baseline(f"bit(oversample={oversample})",
            "CREATE VIRTUAL TABLE vec_items USING vec0("
            "  id integer primary key,"
            "  embedding float[768] distance_metric=cosine,"
            "  embedding_bq bit[768])",
            "INSERT INTO vec_items(id, embedding, embedding_bq) "
            "SELECT id, vector, vec_quantize_binary(vector) "
            "FROM base.train WHERE id >= :lo AND id < :hi",
            lambda c, q, k, os=oversample: c.execute(
                "WITH coarse AS ("
                "  SELECT id, embedding FROM vec_items"
                "  WHERE embedding_bq MATCH vec_quantize_binary(:q)"
                "  LIMIT :ok"
                ") SELECT id, vec_distance_cosine(embedding, :q) as distance "
                "FROM coarse ORDER BY 2 LIMIT :k",
                {"q": q, "k": k, "ok": k * os}).fetchall(),
            out_dir))

    # IVF configs
    ivf_configs = [
        # (nlist, nprobe, train_multiplier)
        (128,  8,  64),
        (128, 16,  64),
        (128, 32,  64),
        (nlist_sqrt,  8,  32),   # sqrt(N)=316, 32 vecs/centroid = 10112 training
        (nlist_sqrt, 16,  32),
        (nlist_sqrt, 32,  32),
    ]
    for nlist, nprobe, train_mult in ivf_configs:
        train_size = min(N, train_mult * nlist)
        name = f"ivf(nlist={nlist},nprobe={nprobe})"
        results.append(run_ivf(name, nlist, nprobe, train_size, out_dir))

    # Final table
    print(f"\n{'='*85}")
    print(f"RESULTS — {N} vectors, COHERE 768-dim cosine, k={K}")
    print(f"{'='*85}")
    print(f"{'name':>30} {'insert':>8} {'train':>7} {'MB':>5} {'qry(ms)':>8} {'recall':>8}")
    print("-" * 75)
    for r in results:
        t = f"{r['train_s']:.0f}s" if r["train_s"] > 0 else "-"
        print(f"{r['name']:>30} {r['insert_s']:>7.1f}s {t:>7} {r['file_mb']:>5.0f} "
              f"{r['mean_ms']:>8.1f} {r['recall']:>8.4f}")


if __name__ == "__main__":
    main()
