#!/usr/bin/env python3
"""Benchmark runner for sqlite-vec KNN configurations.

Measures insert time, build/train time, DB size, KNN latency, and recall
across different vec0 configurations.

Config format: name:type=<index_type>,key=val,key=val

  Baseline (brute-force) keys:
    type=baseline, variant=float|int8|bit, oversample=8

  Index-specific types can be registered via INDEX_REGISTRY (see below).

Usage:
  python bench.py --subset-size 10000 \
    "brute-float:type=baseline,variant=float" \
    "brute-int8:type=baseline,variant=int8" \
    "brute-bit:type=baseline,variant=bit"
"""
import argparse
import json
import os
import sqlite3
import statistics
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_PATH = os.path.join(_SCRIPT_DIR, "..", "dist", "vec0")
INSERT_BATCH_SIZE = 1000

def _discover_datasets():
    """Find datasets: each subdirectory of _SCRIPT_DIR containing base.db."""
    datasets = {}
    for entry in os.listdir(_SCRIPT_DIR):
        base_db = os.path.join(_SCRIPT_DIR, entry, "base.db")
        if os.path.isdir(os.path.join(_SCRIPT_DIR, entry)) and os.path.exists(base_db):
            datasets[entry] = {"base_db": base_db}
    return datasets


DATASETS = _discover_datasets()


# ============================================================================
# Timing helpers
# ============================================================================


def now_ns():
    return time.time_ns()


def ns_to_s(ns):
    return ns / 1_000_000_000


def ns_to_ms(ns):
    return ns / 1_000_000


# ============================================================================
# Index registry — extension point for ANN index branches
# ============================================================================
#
# Each index type provides a dict with:
#   "defaults":          dict of default params
#   "create_table_sql":  fn(params) -> SQL string
#   "insert_sql":        fn(params) -> SQL string  (or None for default)
#   "post_insert_hook":  fn(conn, params) -> train_time_s  (or None)
#   "train_sql":         fn(params) -> SQL string  (or None if no training)
#   "run_query":         fn(conn, params, query, k) -> [(id, distance), ...]  (or None for default MATCH)
#   "query_sql":         fn(params) -> SQL string  (or None for default MATCH)
#   "describe":          fn(params) -> str  (one-line description)
#
# To add a new index type, add an entry here. Example (in your branch):
#
#   INDEX_REGISTRY["diskann"] = {
#       "defaults": {"R": 72, "L": 128, "quantizer": "binary", "buffer_threshold": 0},
#       "create_table_sql": lambda p: f"CREATE VIRTUAL TABLE vec_items USING vec0(...)",
#       "insert_sql": None,
#       "post_insert_hook": None,
#       "run_query": None,
#       "describe": lambda p: f"diskann  q={p['quantizer']}  R={p['R']}  L={p['L']}",
#   }

INDEX_REGISTRY = {}


# ============================================================================
# Baseline implementation
# ============================================================================


def _baseline_create_table_sql(params):
    variant = params["variant"]
    extra = ""
    if variant == "int8":
        extra = ", embedding_int8 int8[768]"
    elif variant == "bit":
        extra = ", embedding_bq bit[768]"
    return (
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  chunk_size=256,"
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"  {extra})"
    )


def _baseline_insert_sql(params):
    variant = params["variant"]
    if variant == "int8":
        return (
            "INSERT INTO vec_items(id, embedding, embedding_int8) "
            "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
            "FROM base.train WHERE id >= :lo AND id < :hi"
        )
    elif variant == "bit":
        return (
            "INSERT INTO vec_items(id, embedding, embedding_bq) "
            "SELECT id, vector, vec_quantize_binary(vector) "
            "FROM base.train WHERE id >= :lo AND id < :hi"
        )
    return None  # use default


def _baseline_run_query(conn, params, query, k):
    variant = params["variant"]
    oversample = params.get("oversample", 8)

    if variant == "int8":
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
    elif variant == "bit":
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

    return None  # use default MATCH


def _baseline_query_sql(params):
    variant = params["variant"]
    oversample = params.get("oversample", 8)
    if variant == "int8":
        return (
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_int8 MATCH vec_quantize_int8(:query, 'unit')"
            f"  LIMIT :k * {oversample}"
            ") "
            "SELECT id, vec_distance_cosine(embedding, :query) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k"
        )
    elif variant == "bit":
        return (
            "WITH coarse AS ("
            "  SELECT id, embedding FROM vec_items"
            "  WHERE embedding_bq MATCH vec_quantize_binary(:query)"
            f"  LIMIT :k * {oversample}"
            ") "
            "SELECT id, vec_distance_cosine(embedding, :query) as distance "
            "FROM coarse ORDER BY 2 LIMIT :k"
        )
    return None


def _baseline_describe(params):
    v = params["variant"]
    if v in ("int8", "bit"):
        return f"baseline  {v} (os={params['oversample']})"
    return f"baseline  {v}"


INDEX_REGISTRY["baseline"] = {
    "defaults": {"variant": "float", "oversample": 8},
    "create_table_sql": _baseline_create_table_sql,
    "insert_sql": _baseline_insert_sql,
    "post_insert_hook": None,
    "train_sql": None,
    "run_query": _baseline_run_query,
    "query_sql": _baseline_query_sql,
    "describe": _baseline_describe,
}


# ============================================================================
# Quantized-only implementation (no rescoring)
# ============================================================================


def _quantized_create_table_sql(params):
    quantizer = params["quantizer"]
    if quantizer == "int8":
        col = "embedding int8[768]"
    elif quantizer == "bit":
        col = "embedding bit[768]"
    else:
        raise ValueError(f"Unknown quantizer: {quantizer}")
    return (
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  chunk_size=256,"
        f"  id integer primary key,"
        f"  {col})"
    )


def _quantized_insert_sql(params):
    quantizer = params["quantizer"]
    if quantizer == "int8":
        return (
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vec_quantize_int8(vector, 'unit') "
            "FROM base.train WHERE id >= :lo AND id < :hi"
        )
    elif quantizer == "bit":
        return (
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vec_quantize_binary(vector) "
            "FROM base.train WHERE id >= :lo AND id < :hi"
        )
    return None


def _quantized_run_query(conn, params, query, k):
    """Search quantized column only — no rescoring."""
    quantizer = params["quantizer"]
    if quantizer == "int8":
        return conn.execute(
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH vec_quantize_int8(:query, 'unit') AND k = :k",
            {"query": query, "k": k},
        ).fetchall()
    elif quantizer == "bit":
        return conn.execute(
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH vec_quantize_binary(:query) AND k = :k",
            {"query": query, "k": k},
        ).fetchall()
    return None


def _quantized_query_sql(params):
    quantizer = params["quantizer"]
    if quantizer == "int8":
        return (
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH vec_quantize_int8(:query, 'unit') AND k = :k"
        )
    elif quantizer == "bit":
        return (
            "SELECT id, distance FROM vec_items "
            "WHERE embedding MATCH vec_quantize_binary(:query) AND k = :k"
        )
    return None


def _quantized_describe(params):
    return f"quantized  {params['quantizer']}"


INDEX_REGISTRY["quantized"] = {
    "defaults": {"quantizer": "bit"},
    "create_table_sql": _quantized_create_table_sql,
    "insert_sql": _quantized_insert_sql,
    "post_insert_hook": None,
    "train_sql": None,
    "run_query": _quantized_run_query,
    "query_sql": _quantized_query_sql,
    "describe": _quantized_describe,
}


# ============================================================================
# Rescore implementation
# ============================================================================


def _rescore_create_table_sql(params):
    quantizer = params.get("quantizer", "bit")
    oversample = params.get("oversample", 8)
    return (
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  chunk_size=256,"
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"  indexed by rescore(quantizer={quantizer}, oversample={oversample}))"
    )


def _rescore_describe(params):
    q = params.get("quantizer", "bit")
    os = params.get("oversample", 8)
    return f"rescore  {q} (os={os})"


INDEX_REGISTRY["rescore"] = {
    "defaults": {"quantizer": "bit", "oversample": 8},
    "create_table_sql": _rescore_create_table_sql,
    "insert_sql": None,
    "post_insert_hook": None,
    "train_sql": None,
    "run_query": None,  # default MATCH query works — rescore is automatic
    "query_sql": None,
    "describe": _rescore_describe,
}


# ============================================================================
# IVF implementation
# ============================================================================


def _ivf_create_table_sql(params):
    return (
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    indexed by ivf("
        f"      nlist={params['nlist']},"
        f"      nprobe={params['nprobe']}"
        f"    )"
        f")"
    )


def _ivf_post_insert_hook(conn, params):
    print("  Training k-means centroids...", flush=True)
    t0 = time.perf_counter()
    conn.execute("INSERT INTO vec_items(id) VALUES ('compute-centroids')")
    conn.commit()
    elapsed = time.perf_counter() - t0
    print(f"  Training done in {elapsed:.1f}s", flush=True)
    return elapsed


def _ivf_describe(params):
    return f"ivf  nlist={params['nlist']:<4} nprobe={params['nprobe']}"


INDEX_REGISTRY["ivf"] = {
    "defaults": {"nlist": 128, "nprobe": 16},
    "create_table_sql": _ivf_create_table_sql,
    "insert_sql": None,
    "post_insert_hook": _ivf_post_insert_hook,
    "train_sql": lambda _: "INSERT INTO vec_items(id) VALUES ('compute-centroids')",
    "run_query": None,
    "query_sql": None,
    "describe": _ivf_describe,
}


# ============================================================================
# DiskANN implementation
# ============================================================================


def _diskann_create_table_sql(params):
    bt = params["buffer_threshold"]
    extra = f", buffer_threshold={bt}" if bt > 0 else ""
    return (
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    INDEXED BY diskann("
        f"      neighbor_quantizer={params['quantizer']},"
        f"      n_neighbors={params['R']},"
        f"      search_list_size={params['L']}"
        f"      {extra}"
        f"    )"
        f")"
    )


def _diskann_pre_query_hook(conn, params):
    L_search = params.get("L_search")
    if L_search:
        conn.execute(
            "INSERT INTO vec_items(id) VALUES (?)",
            (f"search_list_size_search={L_search}",),
        )
        conn.commit()
        print(f"  Set search_list_size_search={L_search}")


def _diskann_describe(params):
    desc = f"diskann  q={params['quantizer']:<6} R={params['R']:<3} L={params['L']}"
    L_search = params.get("L_search")
    if L_search:
        desc += f"  L_search={L_search}"
    return desc


INDEX_REGISTRY["diskann"] = {
    "defaults": {"R": 72, "L": 128, "quantizer": "binary", "buffer_threshold": 0},
    "create_table_sql": _diskann_create_table_sql,
    "insert_sql": None,
    "post_insert_hook": None,
    "pre_query_hook": _diskann_pre_query_hook,
    "train_sql": None,
    "run_query": None,
    "query_sql": None,
    "describe": _diskann_describe,
}


# ============================================================================
# Config parsing
# ============================================================================

INT_KEYS = {
    "R", "L", "L_search", "buffer_threshold", "nlist", "nprobe", "oversample",
    "n_trees", "search_k",
}


def parse_config(spec):
    """Parse 'name:type=baseline,key=val,...' into (name, params_dict)."""
    if ":" in spec:
        name, opts_str = spec.split(":", 1)
    else:
        name, opts_str = spec, ""

    raw = {}
    if opts_str:
        for kv in opts_str.split(","):
            k, v = kv.split("=", 1)
            raw[k.strip()] = v.strip()

    index_type = raw.pop("type", "baseline")
    if index_type not in INDEX_REGISTRY:
        raise ValueError(
            f"Unknown index type: {index_type}. "
            f"Available: {', '.join(sorted(INDEX_REGISTRY.keys()))}"
        )

    reg = INDEX_REGISTRY[index_type]
    params = dict(reg["defaults"])
    for k, v in raw.items():
        if k in INT_KEYS:
            params[k] = int(v)
        else:
            params[k] = v
    params["index_type"] = index_type

    return name, params


def params_to_json(params):
    """Serialize params to JSON, excluding the internal 'index_type' key."""
    return json.dumps({k: v for k, v in sorted(params.items()) if k != "index_type"})


# ============================================================================
# Shared helpers
# ============================================================================


def load_query_vectors(base_db_path, n):
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n", {"n": n}
    ).fetchall()
    conn.close()
    return [(r[0], r[1]) for r in rows]


def insert_loop(conn, sql, subset_size, label="", results_db=None, run_id=None):
    loop_start_ns = now_ns()
    for lo in range(0, subset_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, subset_size)
        batch_start_ns = now_ns()
        conn.execute(sql, {"lo": lo, "hi": hi})
        conn.commit()
        batch_end_ns = now_ns()
        done = hi

        if results_db is not None and run_id is not None:
            elapsed_total_ns = batch_end_ns - loop_start_ns
            elapsed_total_s = ns_to_s(elapsed_total_ns)
            rate = done / elapsed_total_s if elapsed_total_s > 0 else 0
            results_db.execute(
                "INSERT INTO insert_batches "
                "(run_id, batch_lo, batch_hi, rows_in_batch, "
                " started_ns, ended_ns, duration_ns, "
                " cumulative_rows, rate_rows_per_s) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id, lo, hi, hi - lo,
                    batch_start_ns, batch_end_ns,
                    batch_end_ns - batch_start_ns,
                    done, round(rate, 1),
                ),
            )

        if done % 5000 == 0 or done == subset_size:
            elapsed_total_ns = batch_end_ns - loop_start_ns
            elapsed_total_s = ns_to_s(elapsed_total_ns)
            rate = done / elapsed_total_s if elapsed_total_s > 0 else 0
            print(
                f"    [{label}] {done:>8}/{subset_size}  "
                f"{elapsed_total_s:.1f}s  {rate:.0f} rows/s",
                flush=True,
            )
            if results_db is not None:
                results_db.commit()

    return time.perf_counter()  # not used for timing anymore, kept for compat


def open_bench_db(db_path, ext_path, base_db):
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")
    return conn


DEFAULT_INSERT_SQL = (
    "INSERT INTO vec_items(id, embedding) "
    "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi"
)

DEFAULT_QUERY_SQL = (
    "SELECT id, distance FROM vec_items "
    "WHERE embedding MATCH :query AND k = :k"
)


# ============================================================================
# Results DB helpers
# ============================================================================

_RESULTS_SCHEMA_PATH = os.path.join(_SCRIPT_DIR, "results_schema.sql")


def open_results_db(out_dir, dataset, subset_size):
    """Open/create the results DB in WAL mode."""
    sub_dir = os.path.join(out_dir, dataset, str(subset_size))
    os.makedirs(sub_dir, exist_ok=True)
    db_path = os.path.join(sub_dir, "results.db")
    db = sqlite3.connect(db_path, timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=30000")
    with open(_RESULTS_SCHEMA_PATH) as f:
        db.executescript(f.read())
    return db, sub_dir


def create_run(results_db, config_name, index_type, params, dataset,
               subset_size, k, n_queries):
    """Insert a new run row and return the run_id."""
    cur = results_db.execute(
        "INSERT INTO runs "
        "(config_name, index_type, params, dataset, subset_size, "
        " k, n_queries, status, created_at_ns) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            config_name, index_type, params_to_json(params), dataset,
            subset_size, k, n_queries, "pending", now_ns(),
        ),
    )
    results_db.commit()
    return cur.lastrowid


def update_run_status(results_db, run_id, status):
    results_db.execute(
        "UPDATE runs SET status=? WHERE run_id=?", (status, run_id)
    )
    results_db.commit()


# ============================================================================
# Build
# ============================================================================


def build_index(base_db, ext_path, name, params, subset_size, sub_dir,
                results_db=None, run_id=None, k=None):
    db_path = os.path.join(sub_dir, f"{name}.{subset_size}.db")
    conn = open_bench_db(db_path, ext_path, base_db)

    reg = INDEX_REGISTRY[params["index_type"]]

    create_sql = reg["create_table_sql"](params)
    conn.execute(create_sql)

    label = params["index_type"]
    print(f"  Inserting {subset_size} vectors...")

    sql_fn = reg.get("insert_sql")
    insert_sql = sql_fn(params) if sql_fn else None
    if insert_sql is None:
        insert_sql = DEFAULT_INSERT_SQL

    train_sql_fn = reg.get("train_sql")
    train_sql = train_sql_fn(params) if train_sql_fn else None

    query_sql_fn = reg.get("query_sql")
    query_sql = query_sql_fn(params) if query_sql_fn else None
    if query_sql is None:
        query_sql = DEFAULT_QUERY_SQL

    # -- Insert phase --
    if results_db and run_id:
        update_run_status(results_db, run_id, "inserting")
    insert_started_ns = now_ns()

    insert_loop(conn, insert_sql, subset_size, label,
                results_db=results_db, run_id=run_id)

    insert_ended_ns = now_ns()
    insert_duration_ns = insert_ended_ns - insert_started_ns

    # -- Training phase --
    train_started_ns = None
    train_ended_ns = None
    train_duration_ns = None
    train_time_s = 0.0
    hook = reg.get("post_insert_hook")
    if hook:
        if results_db and run_id:
            update_run_status(results_db, run_id, "training")
        train_started_ns = now_ns()
        train_time_s = hook(conn, params)
        train_ended_ns = now_ns()
        train_duration_ns = train_ended_ns - train_started_ns

    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size_bytes = os.path.getsize(db_path)

    build_duration_ns = insert_duration_ns + (train_duration_ns or 0)
    insert_time_s = ns_to_s(insert_duration_ns)

    # Write run_results (build portion)
    if results_db and run_id:
        results_db.execute(
            "INSERT INTO run_results "
            "(run_id, insert_started_ns, insert_ended_ns, insert_duration_ns, "
            " train_started_ns, train_ended_ns, train_duration_ns, "
            " build_duration_ns, db_file_size_bytes, db_file_path, "
            " create_sql, insert_sql, train_sql, query_sql, k) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, insert_started_ns, insert_ended_ns, insert_duration_ns,
                train_started_ns, train_ended_ns, train_duration_ns,
                build_duration_ns, file_size_bytes, db_path,
                create_sql, insert_sql, train_sql, query_sql, k,
            ),
        )
        results_db.commit()

    return {
        "db_path": db_path,
        "insert_time_s": round(insert_time_s, 3),
        "train_time_s": round(train_time_s, 3),
        "total_time_s": round(insert_time_s + train_time_s, 3),
        "insert_per_vec_ms": round((insert_time_s / row_count) * 1000, 2)
        if row_count
        else 0,
        "rows": row_count,
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
    }


# ============================================================================
# KNN measurement
# ============================================================================


def _default_match_query(conn, query, k):
    return conn.execute(
        "SELECT id, distance FROM vec_items "
        "WHERE embedding MATCH :query AND k = :k",
        {"query": query, "k": k},
    ).fetchall()


def measure_knn(db_path, ext_path, base_db, params, subset_size, k=10, n=50,
                results_db=None, run_id=None, pre_query_hook=None):
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    if pre_query_hook:
        pre_query_hook(conn, params)

    query_vectors = load_query_vectors(base_db, n)

    reg = INDEX_REGISTRY[params["index_type"]]
    query_fn = reg.get("run_query")

    if results_db and run_id:
        update_run_status(results_db, run_id, "querying")

    times_ms = []
    recalls = []
    for i, (qid, query) in enumerate(query_vectors):
        started_ns = now_ns()

        results = None
        if query_fn:
            results = query_fn(conn, params, query, k)
        if results is None:
            results = _default_match_query(conn, query, k)

        ended_ns = now_ns()
        duration_ms = ns_to_ms(ended_ns - started_ns)
        times_ms.append(duration_ms)

        result_ids_list = [r[0] for r in results]
        result_distances_list = [r[1] for r in results]
        result_ids = set(result_ids_list)

        # Ground truth: use pre-computed neighbors table for full dataset,
        # otherwise brute-force over the subset
        if subset_size >= 1000000:
            gt_rows = conn.execute(
                "SELECT CAST(neighbors_id AS INTEGER) FROM base.neighbors "
                "WHERE query_vector_id = :qid AND rank < :k",
                {"qid": qid, "k": k},
            ).fetchall()
        else:
            gt_rows = conn.execute(
                "SELECT id FROM ("
                "  SELECT id, vec_distance_cosine(vector, :query) as dist "
                "  FROM base.train WHERE id < :n ORDER BY dist LIMIT :k"
                ")",
                {"query": query, "k": k, "n": subset_size},
            ).fetchall()
        gt_ids_list = [r[0] for r in gt_rows]
        gt_ids = set(gt_ids_list)

        if gt_ids:
            q_recall = len(result_ids & gt_ids) / len(gt_ids)
        else:
            q_recall = 0.0
        recalls.append(q_recall)

        if results_db and run_id:
            results_db.execute(
                "INSERT INTO queries "
                "(run_id, k, query_vector_id, started_ns, ended_ns, duration_ms, "
                " result_ids, result_distances, ground_truth_ids, recall) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, k, qid, started_ns, ended_ns, round(duration_ms, 4),
                    json.dumps(result_ids_list),
                    json.dumps(result_distances_list),
                    json.dumps(gt_ids_list),
                    round(q_recall, 6),
                ),
            )
            if (i + 1) % 10 == 0 or (i + 1) == len(query_vectors):
                results_db.commit()

    conn.close()

    mean_ms = round(statistics.mean(times_ms), 2)
    median_ms = round(statistics.median(times_ms), 2)
    p99_ms = (round(sorted(times_ms)[int(len(times_ms) * 0.99)], 2)
              if len(times_ms) > 1
              else round(times_ms[0], 2))
    total_ms = round(sum(times_ms), 2)
    recall = round(statistics.mean(recalls), 4)
    qps = round(len(times_ms) / (total_ms / 1000), 1) if total_ms > 0 else 0

    # Update run_results with query aggregates
    if results_db and run_id:
        results_db.execute(
            "UPDATE run_results SET "
            "query_mean_ms=?, query_median_ms=?, query_p99_ms=?, "
            "query_total_ms=?, qps=?, recall=? "
            "WHERE run_id=?",
            (mean_ms, median_ms, p99_ms, total_ms, qps, recall, run_id),
        )
        update_run_status(results_db, run_id, "done")

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p99_ms": p99_ms,
        "total_ms": total_ms,
        "recall": recall,
    }


# ============================================================================
# Reporting
# ============================================================================


def print_report(all_results):
    print(
        f"\n{'name':>20} {'N':>7} {'type':>10} {'config':>28}  "
        f"{'ins(s)':>7} {'train':>6} {'MB':>7} "
        f"{'qry(ms)':>8} {'recall':>7}"
    )
    print("-" * 115)
    for r in all_results:
        train = f"{r['train_time_s']:.1f}" if r["train_time_s"] > 0 else "-"
        print(
            f"{r['name']:>20} {r['n_vectors']:>7} {r['index_type']:>10} "
            f"{r['config_desc']:>28}  "
            f"{r['insert_time_s']:>7.1f} {train:>6} {r['file_size_mb']:>7.1f} "
            f"{r['mean_ms']:>8.2f} {r['recall']:>7.4f}"
        )


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark runner for sqlite-vec KNN configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("configs", nargs="+", help="config specs (name:type=X,key=val,...)")
    parser.add_argument("--subset-size", type=int, required=True)
    parser.add_argument("-k", type=int, default=10, help="KNN k (default 10)")
    parser.add_argument("-n", type=int, default=50, help="number of queries (default 50)")
    parser.add_argument("--dataset", default="cohere1m",
                        choices=list(DATASETS.keys()),
                        help="dataset name (default: cohere1m)")
    parser.add_argument("--ext", default=EXT_PATH)
    parser.add_argument("-o", "--out-dir", default="runs")
    args = parser.parse_args()

    dataset_cfg = DATASETS[args.dataset]
    base_db = dataset_cfg["base_db"]

    results_db, sub_dir = open_results_db(args.out_dir, args.dataset, args.subset_size)
    configs = [parse_config(c) for c in args.configs]

    all_results = []
    for i, (name, params) in enumerate(configs, 1):
        reg = INDEX_REGISTRY[params["index_type"]]
        desc = reg["describe"](params)
        print(f"\n[{i}/{len(configs)}] {name}  ({desc.strip()})")

        run_id = create_run(
            results_db, name, params["index_type"], params,
            args.dataset, args.subset_size, args.k, args.n,
        )

        try:
            build = build_index(
                base_db, args.ext, name, params, args.subset_size, sub_dir,
                results_db=results_db, run_id=run_id, k=args.k,
            )
            train_str = f" + {build['train_time_s']}s train" if build["train_time_s"] > 0 else ""
            print(
                f"  Build: {build['insert_time_s']}s insert{train_str}  "
                f"{build['file_size_mb']} MB"
            )

            pre_hook = reg.get("pre_query_hook")
            print(f"  Measuring KNN (k={args.k}, n={args.n})...")
            knn = measure_knn(
                build["db_path"], args.ext, base_db,
                params, args.subset_size, k=args.k, n=args.n,
                results_db=results_db, run_id=run_id,
                pre_query_hook=pre_hook,
            )
            print(f"  KNN: mean={knn['mean_ms']}ms  recall@{args.k}={knn['recall']}")
        except Exception as e:
            update_run_status(results_db, run_id, "error")
            print(f"  ERROR: {e}")
            raise

        all_results.append({
            "name": name,
            "n_vectors": args.subset_size,
            "index_type": params["index_type"],
            "config_desc": desc,
            "db_path": build["db_path"],
            "insert_time_s": build["insert_time_s"],
            "train_time_s": build["train_time_s"],
            "total_time_s": build["total_time_s"],
            "insert_per_vec_ms": build["insert_per_vec_ms"],
            "rows": build["rows"],
            "file_size_mb": build["file_size_mb"],
            "k": args.k,
            "n_queries": args.n,
            "mean_ms": knn["mean_ms"],
            "median_ms": knn["median_ms"],
            "p99_ms": knn["p99_ms"],
            "total_ms": knn["total_ms"],
            "recall": knn["recall"],
        })

    print_report(all_results)

    print(f"\nResults DB: {os.path.join(sub_dir, 'results.db')}")
    results_db.close()


if __name__ == "__main__":
    main()
