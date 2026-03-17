#!/usr/bin/env python3
"""Benchmark IVF quantization: none vs int8 vs binary, with oversample sweep.

Builds each index once, uses runtime nprobe to sweep query params.
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
K = 10
N_QUERIES = 50
NLIST = int(math.sqrt(N))  # 316
TRAIN_MULT = 16


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


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
        "recall": round(statistics.mean(recalls), 4),
    }


def build_ivf(name, nlist, quantizer, oversample, train_size, out_dir):
    """Build an IVF index. Returns (db_path, insert_time, train_time, file_mb)."""
    db_path = os.path.join(out_dir, f"{name}.{N}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

    q_clause = ""
    if quantizer != "none":
        q_clause = f", quantizer={quantizer}"
        if oversample > 1:
            q_clause += f", oversample={oversample}"

    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    indexed by ivf(nlist={nlist}, nprobe=1{q_clause})"
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
    phase1 = time.perf_counter() - t0

    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train

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
    phase3 = time.perf_counter() - t_rest
    total_insert = phase1 + phase3

    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    return db_path, round(total_insert, 1), round(train_time, 1), round(file_mb, 0)


def build_baseline(name, create_sql, insert_sql, out_dir):
    db_path = os.path.join(out_dir, f"{name}.{N}.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")
    conn.execute(create_sql)
    t0 = time.perf_counter()
    for lo in range(0, N, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, N)
        conn.execute(insert_sql, {"lo": lo, "hi": hi})
        conn.commit()
    insert_time = time.perf_counter() - t0
    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    return db_path, round(insert_time, 1), round(file_mb, 0)


def query_nprobe(db_path, nprobe):
    """Set nprobe and measure KNN in the same connection."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")
    conn.execute(f"INSERT INTO vec_items(id) VALUES ('nprobe={nprobe}')")

    queries = load_query_vectors(BASE_DB, N_QUERIES)
    times_ms = []
    recalls = []
    for qid, query in queries:
        t0 = time.perf_counter()
        results = conn.execute(
            "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
            {"q": query, "k": K},
        ).fetchall()
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
        "recall": round(statistics.mean(recalls), 4),
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    train_size = min(N, TRAIN_MULT * NLIST)
    results = []

    print(f"=== IVF Quantization Benchmark: N={N:,}, nlist={NLIST}, train={train_size} ===\n")

    # Baselines
    print("Building baselines...", flush=True)
    flat_path, flat_ins, flat_mb = build_baseline("flat",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine)",
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        out_dir)
    flat_knn = measure_knn(flat_path, lambda c, q, k: c.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
        {"q": q, "k": k}).fetchall())
    results.append({"name": "flat", "insert_s": flat_ins, "train_s": 0,
                     "file_mb": flat_mb, **flat_knn})
    print(f"  flat: {flat_knn['mean_ms']}ms, {flat_knn['recall']} recall, {flat_mb} MB", flush=True)
    os.remove(flat_path)

    int8_path, int8_ins, int8_mb = build_baseline("int8(os=4)",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine,"
        "  embedding_int8 int8[768])",
        "INSERT INTO vec_items(id, embedding, embedding_int8) "
        "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        out_dir)
    int8_knn = measure_knn(int8_path, lambda c, q, k: c.execute(
        "WITH coarse AS (SELECT id, embedding FROM vec_items "
        "WHERE embedding_int8 MATCH vec_quantize_int8(:q, 'unit') LIMIT :ok) "
        "SELECT id, vec_distance_cosine(embedding, :q) as distance "
        "FROM coarse ORDER BY 2 LIMIT :k",
        {"q": q, "k": k, "ok": k * 4}).fetchall())
    results.append({"name": "int8(os=4)", "insert_s": int8_ins, "train_s": 0,
                     "file_mb": int8_mb, **int8_knn})
    print(f"  int8(os=4): {int8_knn['mean_ms']}ms, {int8_knn['recall']} recall", flush=True)
    os.remove(int8_path)

    bit_path, bit_ins, bit_mb = build_baseline("bit(os=8)",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine,"
        "  embedding_bq bit[768])",
        "INSERT INTO vec_items(id, embedding, embedding_bq) "
        "SELECT id, vector, vec_quantize_binary(vector) "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        out_dir)
    bit_knn = measure_knn(bit_path, lambda c, q, k: c.execute(
        "WITH coarse AS (SELECT id, embedding FROM vec_items "
        "WHERE embedding_bq MATCH vec_quantize_binary(:q) LIMIT :ok) "
        "SELECT id, vec_distance_cosine(embedding, :q) as distance "
        "FROM coarse ORDER BY 2 LIMIT :k",
        {"q": q, "k": k, "ok": k * 8}).fetchall())
    results.append({"name": "bit(os=8)", "insert_s": bit_ins, "train_s": 0,
                     "file_mb": bit_mb, **bit_knn})
    print(f"  bit(os=8): {bit_knn['mean_ms']}ms, {bit_knn['recall']} recall", flush=True)
    os.remove(bit_path)

    # IVF configs: (label, quantizer, oversample, nprobes_to_test)
    ivf_configs = [
        ("ivf(q=none)",         "none",   1, [8, 16, 32]),
        ("ivf(q=int8,os=1)",    "int8",   1, [8, 16, 32]),
        ("ivf(q=int8,os=4)",    "int8",   4, [8, 16, 32]),
        ("ivf(q=binary,os=1)",  "binary", 1, [16, 32, 64]),
        ("ivf(q=binary,os=10)", "binary", 10, [16, 32, 64]),
        ("ivf(q=binary,os=50)", "binary", 50, [16, 32, 64]),
    ]

    for label, quantizer, oversample, nprobes in ivf_configs:
        print(f"\n  Building {label}...", flush=True)
        db_path, ins_time, train_time, file_mb = build_ivf(
            label, NLIST, quantizer, oversample, train_size, out_dir)
        print(f"    insert={ins_time}s, train={train_time}s, {file_mb} MB", flush=True)

        for nprobe in nprobes:
            knn = query_nprobe(db_path, nprobe)
            name = f"{label},p={nprobe}"
            results.append({"name": name, "insert_s": ins_time, "train_s": train_time,
                             "file_mb": file_mb, **knn})
            print(f"    nprobe={nprobe}: {knn['mean_ms']}ms, {knn['recall']} recall", flush=True)

        os.remove(db_path)

    # Final table
    print(f"\n\n{'='*80}")
    print(f"RESULTS — {N:,} vectors, COHERE 768-dim cosine, k={K}, nlist={NLIST}")
    print(f"{'='*80}")
    print(f"{'name':>30} {'insert':>7} {'train':>6} {'MB':>5} {'qry(ms)':>8} {'recall':>8}")
    print("-" * 70)
    for r in results:
        t = f"{r['train_s']:.0f}s" if r.get("train_s", 0) > 0 else "-"
        print(f"{r['name']:>30} {r['insert_s']:>6.1f}s {t:>6} {r['file_mb']:>5.0f} "
              f"{r['mean_ms']:>8.1f} {r['recall']:>8.4f}")


if __name__ == "__main__":
    main()
