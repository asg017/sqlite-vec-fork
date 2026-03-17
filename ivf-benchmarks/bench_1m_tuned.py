#!/usr/bin/env python3
"""1M benchmark: best baselines + IVF with tuned params.

Based on 100k tuning results:
- int8(oversample=4): best speed/recall for int8
- bit(oversample=8): best speed/recall for bit
- IVF: nlist=128 and sqrt(N)=1000, nprobe=8,16
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


def insert_loop(conn, sql, total, label=""):
    t0 = time.perf_counter()
    for lo in range(0, total, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, total)
        conn.execute(sql, {"lo": lo, "hi": hi})
        conn.commit()
        done = hi
        if done % 200000 == 0 or done == total:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    [{label}] {done:>8}/{total}  {elapsed:.0f}s  {rate:.0f} rows/s", flush=True)
    return time.perf_counter() - t0


def measure_knn(db_path, query_fn):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")
    queries = load_query_vectors(BASE_DB, N_QUERIES)
    times_ms = []
    recalls = []
    for i, (qid, query) in enumerate(queries):
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
        if (i + 1) % 10 == 0:
            print(f"      query {i+1}/{N_QUERIES}  "
                  f"mean={statistics.mean(times_ms):.1f}ms  "
                  f"recall={statistics.mean(recalls):.3f}", flush=True)
    conn.close()
    return {
        "mean_ms": round(statistics.mean(times_ms), 1),
        "median_ms": round(statistics.median(times_ms), 1),
        "recall": round(statistics.mean(recalls), 4),
    }


def run_baseline(name, create_sql, insert_sql, query_fn, out_dir):
    print(f"\n{'='*60}\n  {name}\n{'='*60}", flush=True)
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
    print(f"  Build: {insert_time:.1f}s  ({file_mb:.0f} MB)", flush=True)
    print(f"  KNN (k={K}, n={N_QUERIES})...", flush=True)
    knn = measure_knn(db_path, query_fn)
    print(f"  => {knn['mean_ms']}ms  recall={knn['recall']}", flush=True)
    os.remove(db_path)
    return {"name": name, "insert_s": round(insert_time, 1), "train_s": 0,
            "file_mb": round(file_mb, 0), **knn}


def run_ivf(name, nlist, nprobe, train_size, out_dir):
    print(f"\n{'='*60}\n  {name} (train={train_size})\n{'='*60}", flush=True)
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

    print(f"  Training k-means (nlist={nlist})...", flush=True)
    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train
    print(f"  Train: {train_time:.1f}s", flush=True)

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
        done = hi - train_size
        if done % 200000 == 0 or hi == N:
            elapsed = time.perf_counter() - t_rest
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{remaining}  {elapsed:.0f}s  {rate:.0f} rows/s", flush=True)
    phase3_time = time.perf_counter() - t_rest
    total_insert = phase1_time + phase3_time

    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"  Build: {total_insert:.1f}s insert + {train_time:.1f}s train  ({file_mb:.0f} MB)", flush=True)

    print(f"  KNN (k={K}, n={N_QUERIES})...", flush=True)
    knn = measure_knn(db_path, lambda c, q, k: c.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
        {"q": q, "k": k}).fetchall())
    print(f"  => {knn['mean_ms']}ms  recall={knn['recall']}", flush=True)
    os.remove(db_path)
    return {"name": name, "insert_s": round(total_insert, 1),
            "train_s": round(train_time, 1), "file_mb": round(file_mb, 0), **knn}


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    results = []

    nlist_sqrt = int(math.sqrt(N))  # 1000

    print(f"=== 1M Benchmark (COHERE 768-dim cosine) ===\n")

    # 3 baselines: flat, best int8, best bit
    results.append(run_baseline("flat",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine)",
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        lambda c, q, k: c.execute(
            "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
            {"q": q, "k": k}).fetchall(),
        out_dir))

    results.append(run_baseline("int8(oversample=4)",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key,"
        "  embedding float[768] distance_metric=cosine,"
        "  embedding_int8 int8[768])",
        "INSERT INTO vec_items(id, embedding, embedding_int8) "
        "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        lambda c, q, k: c.execute(
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_int8 MATCH vec_quantize_int8(:q, 'unit')"
            "  LIMIT :ok"
            ") SELECT id, vec_distance_cosine(embedding, :q) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k",
            {"q": q, "k": k, "ok": k * 4}).fetchall(),
        out_dir))

    results.append(run_baseline("bit(oversample=8)",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key,"
        "  embedding float[768] distance_metric=cosine,"
        "  embedding_bq bit[768])",
        "INSERT INTO vec_items(id, embedding, embedding_bq) "
        "SELECT id, vector, vec_quantize_binary(vector) "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        lambda c, q, k: c.execute(
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_bq MATCH vec_quantize_binary(:q)"
            "  LIMIT :ok"
            ") SELECT id, vec_distance_cosine(embedding, :q) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k",
            {"q": q, "k": k, "ok": k * 8}).fetchall(),
        out_dir))

    # 3 IVF configs: nlist=128/p16, nlist=1000/p8, nlist=1000/p16
    for nlist, nprobe in [(128, 16), (nlist_sqrt, 8), (nlist_sqrt, 16)]:
        train_size = min(N, 64 * nlist)
        name = f"ivf(nlist={nlist},nprobe={nprobe})"
        results.append(run_ivf(name, nlist, nprobe, train_size, out_dir))

    # Final table
    print(f"\n\n{'='*85}")
    print(f"RESULTS — {N:,} vectors, COHERE 768-dim cosine, k={K}")
    print(f"{'='*85}")
    print(f"{'name':>32} {'insert':>8} {'train':>7} {'MB':>6} {'qry(ms)':>8} {'recall':>8}")
    print("-" * 77)
    for r in results:
        t = f"{r['train_s']:.0f}s" if r["train_s"] > 0 else "-"
        print(f"{r['name']:>32} {r['insert_s']:>7.1f}s {t:>7} {r['file_mb']:>6.0f} "
              f"{r['mean_ms']:>8.1f} {r['recall']:>8.4f}")


if __name__ == "__main__":
    main()
