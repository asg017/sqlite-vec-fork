#!/usr/bin/env python3
"""1M IVF quantization benchmark with full build/train/query timings."""
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
NLIST = int(math.sqrt(N))  # 1000
TRAIN_MULT = 16  # 16000 training vectors


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def build_baseline(name, create_sql, insert_sql, out_dir):
    print(f"\n  Building {name}...", flush=True)
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
        done = hi
        if done % 500000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    {done:>8}/{N}  {elapsed:.0f}s  {done/elapsed:.0f} rows/s", flush=True)
    insert_time = time.perf_counter() - t0
    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"    Done: {insert_time:.1f}s, {file_mb:.0f} MB", flush=True)
    return db_path, round(insert_time, 1), round(file_mb, 0)


def build_ivf(name, quantizer, oversample, out_dir):
    train_size = min(N, TRAIN_MULT * NLIST)
    print(f"\n  Building {name} (train={train_size})...", flush=True)
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
        f"    indexed by ivf(nlist={NLIST}, nprobe=1{q_clause})"
        f")"
    )

    # Phase 1: training vectors
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
    print(f"    Phase 1: {train_size} vectors in {phase1:.1f}s", flush=True)

    # Phase 2: train
    t_train = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    train_time = time.perf_counter() - t_train
    print(f"    Phase 2: k-means in {train_time:.1f}s", flush=True)

    # Phase 3: remaining
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
            eta = (remaining - done) / rate if rate > 0 else 0
            print(f"    Phase 3: {done:>8}/{remaining}  {elapsed:.0f}s  {rate:.0f} rows/s  eta {eta:.0f}s", flush=True)
    phase3 = time.perf_counter() - t_rest
    total_insert = phase1 + phase3

    conn.close()
    file_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"    Done: {total_insert:.1f}s insert + {train_time:.1f}s train, {file_mb:.0f} MB", flush=True)
    return db_path, round(total_insert, 1), round(train_time, 1), round(file_mb, 0)


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
        if (i + 1) % 25 == 0:
            print(f"      query {i+1}/{N_QUERIES}  mean={statistics.mean(times_ms):.1f}ms  recall={statistics.mean(recalls):.3f}", flush=True)
    conn.close()
    return {
        "mean_ms": round(statistics.mean(times_ms), 1),
        "recall": round(statistics.mean(recalls), 4),
    }


def query_ivf(db_path, nprobe):
    """Set nprobe and measure in same connection."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(EXT_PATH)
    conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")
    conn.execute(f"INSERT INTO vec_items(id) VALUES ('nprobe={nprobe}')")
    queries = load_query_vectors(BASE_DB, N_QUERIES)
    times_ms = []
    recalls = []
    for i, (qid, query) in enumerate(queries):
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
        if (i + 1) % 25 == 0:
            print(f"      query {i+1}/{N_QUERIES}  mean={statistics.mean(times_ms):.1f}ms  recall={statistics.mean(recalls):.3f}", flush=True)
    conn.close()
    return {
        "mean_ms": round(statistics.mean(times_ms), 1),
        "recall": round(statistics.mean(recalls), 4),
    }


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(out_dir, exist_ok=True)
    results = []

    print(f"=== 1M Quantization Benchmark: nlist={NLIST}, train={TRAIN_MULT}x ===\n")

    # Baselines
    flat_path, flat_ins, flat_mb = build_baseline("flat",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine)",
        "INSERT INTO vec_items(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        out_dir)
    print("    Querying flat...", flush=True)
    flat_knn = measure_knn(flat_path, lambda c, q, k: c.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH :q AND k=:k",
        {"q": q, "k": k}).fetchall())
    results.append({"name": "flat", "insert_s": flat_ins, "train_s": 0, "file_mb": flat_mb, **flat_knn})
    print(f"    => {flat_knn['mean_ms']}ms, {flat_knn['recall']} recall", flush=True)
    os.remove(flat_path)

    int8_path, int8_ins, int8_mb = build_baseline("int8(os=4)",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine,"
        "  embedding_int8 int8[768])",
        "INSERT INTO vec_items(id, embedding, embedding_int8) "
        "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        out_dir)
    print("    Querying int8...", flush=True)
    int8_knn = measure_knn(int8_path, lambda c, q, k: c.execute(
        "WITH coarse AS (SELECT id, embedding FROM vec_items "
        "WHERE embedding_int8 MATCH vec_quantize_int8(:q, 'unit') LIMIT :ok) "
        "SELECT id, vec_distance_cosine(embedding, :q) as distance "
        "FROM coarse ORDER BY 2 LIMIT :k",
        {"q": q, "k": k, "ok": k * 4}).fetchall())
    results.append({"name": "int8(os=4)", "insert_s": int8_ins, "train_s": 0, "file_mb": int8_mb, **int8_knn})
    print(f"    => {int8_knn['mean_ms']}ms, {int8_knn['recall']} recall", flush=True)
    os.remove(int8_path)

    bit_path, bit_ins, bit_mb = build_baseline("bit(os=8)",
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "  id integer primary key, embedding float[768] distance_metric=cosine,"
        "  embedding_bq bit[768])",
        "INSERT INTO vec_items(id, embedding, embedding_bq) "
        "SELECT id, vector, vec_quantize_binary(vector) "
        "FROM base.train WHERE id >= :lo AND id < :hi",
        out_dir)
    print("    Querying bit...", flush=True)
    bit_knn = measure_knn(bit_path, lambda c, q, k: c.execute(
        "WITH coarse AS (SELECT id, embedding FROM vec_items "
        "WHERE embedding_bq MATCH vec_quantize_binary(:q) LIMIT :ok) "
        "SELECT id, vec_distance_cosine(embedding, :q) as distance "
        "FROM coarse ORDER BY 2 LIMIT :k",
        {"q": q, "k": k, "ok": k * 8}).fetchall())
    results.append({"name": "bit(os=8)", "insert_s": bit_ins, "train_s": 0, "file_mb": bit_mb, **bit_knn})
    print(f"    => {bit_knn['mean_ms']}ms, {bit_knn['recall']} recall", flush=True)
    os.remove(bit_path)

    # IVF configs: (label, quantizer, oversample, [nprobes])
    ivf_configs = [
        ("ivf(q=none)",         "none",    1, [8, 16, 32]),
        ("ivf(q=int8,os=4)",    "int8",    4, [8, 16, 32]),
        ("ivf(q=binary,os=10)", "binary", 10, [16, 32, 64]),
        ("ivf(q=binary,os=50)", "binary", 50, [32, 64]),
    ]

    for label, quantizer, oversample, nprobes in ivf_configs:
        db_path, ins_time, train_time, file_mb = build_ivf(label, quantizer, oversample, out_dir)

        for nprobe in nprobes:
            print(f"    Querying {label} nprobe={nprobe}...", flush=True)
            knn = query_ivf(db_path, nprobe)
            name = f"{label},p={nprobe}"
            results.append({"name": name, "insert_s": ins_time, "train_s": train_time,
                             "file_mb": file_mb, **knn})
            print(f"    => {knn['mean_ms']}ms, {knn['recall']} recall", flush=True)

        os.remove(db_path)

    # Final table
    print(f"\n\n{'='*85}")
    print(f"RESULTS — {N:,} vectors, COHERE 768-dim cosine, k={K}, nlist={NLIST}")
    print(f"{'='*85}")
    print(f"{'name':>32} {'insert':>7} {'train':>6} {'MB':>6} {'qry(ms)':>8} {'recall':>8}")
    print("-" * 75)
    for r in results:
        t = f"{r['train_s']:.0f}s" if r.get("train_s", 0) > 0 else "-"
        print(f"{r['name']:>32} {r['insert_s']:>6.1f}s {t:>6} {r['file_mb']:>6.0f} "
              f"{r['mean_ms']:>8.1f} {r['recall']:>8.4f}")


if __name__ == "__main__":
    main()
