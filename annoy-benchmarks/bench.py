#!/usr/bin/env python3
"""Benchmark Annoy index configurations vs brute-force baselines.

Compares insert time, build time, DB size, KNN query time, and recall
across Annoy configs and baseline vec0 tables (float, int8, bit).

Usage:
  # Annoy vs baselines at 10k
  python3 bench.py --subset-size 10000 \
      --configs \
          "annoy-t10:n_trees=10" \
          "annoy-t50:n_trees=50" \
          "annoy-t100:n_trees=100" \
          "brute-float:type=float" \
          "brute-int8:type=int8" \
          "brute-bit:type=bit" \
      -k 10 -n 50 -o runs/10k

Config format: name:key=val,key=val
  Annoy keys: n_trees, search_k
  Baseline keys: type (float|int8|bit), oversample (for int8/bit rescore)
  If 'type' is set, it's a baseline. Otherwise it's Annoy.
"""
import argparse
import os
import sqlite3
import statistics
import sys
import time
import struct


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_PATH = os.path.join(_SCRIPT_DIR, "..", "dist", "vec0")
BASE_DB = os.path.join(_SCRIPT_DIR, "seed", "base.db")
INSERT_BATCH_SIZE = 1000

ANNOY_DEFAULTS = {
    "n_trees": 50,
    "search_k": 0,
    "quantizer": "none",
}


def load_query_vectors(base_db_path, n):
    """Load query vector ids and raw bytes from base.db."""
    conn = sqlite3.connect(base_db_path)
    rows = conn.execute(
        "SELECT id, vector FROM query_vectors ORDER BY id LIMIT :n",
        {"n": n},
    ).fetchall()
    conn.close()
    return [(row[0], row[1]) for row in rows]


def parse_config(spec):
    """Parse 'name:key=val,key=val' into (name, params_dict)."""
    if ":" in spec:
        name, opts_str = spec.split(":", 1)
    else:
        name, opts_str = spec, ""

    params = {}
    if opts_str:
        for kv in opts_str.split(","):
            k, v = kv.split("=", 1)
            params[k.strip()] = v.strip()

    if "type" in params:
        btype = params["type"]
        if btype not in ("float", "int8", "bit"):
            raise argparse.ArgumentTypeError(
                f"Unknown baseline type: {btype}. Use float, int8, or bit.")
        oversample = int(params.get("oversample", "8"))
        return name, {
            "index_type": "baseline",
            "baseline_type": btype,
            "oversample": oversample,
        }
    else:
        cfg = dict(ANNOY_DEFAULTS)
        for k, v in params.items():
            if k in ("n_trees", "search_k"):
                cfg[k] = int(v)
            elif k == "quantizer":
                if v not in ("none", "int8", "binary"):
                    raise argparse.ArgumentTypeError(f"Unknown quantizer: {v}")
                cfg[k] = v
            else:
                raise argparse.ArgumentTypeError(f"Unknown Annoy key: {k}")
        cfg["index_type"] = "annoy"
        return name, cfg


# ---------- Build ----------

def build_annoy(base_db, ext_path, name, params, subset_size, out_dir):
    db_path = os.path.join(out_dir, f"{name}.{subset_size}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    extra_opts = ""
    if params["search_k"] > 0:
        extra_opts += f", search_k={params['search_k']}"
    if params["quantizer"] != "none":
        extra_opts += f", quantizer={params['quantizer']}"

    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    INDEXED BY annoy("
        f"      n_trees={params['n_trees']}"
        f"      {extra_opts}"
        f"    )"
        f")"
    )

    # Phase 1: Insert all vectors
    print(f"  Inserting {subset_size} vectors...")
    t0 = time.perf_counter()
    for lo in range(0, subset_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, subset_size)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()
        done = hi
        if done % 5000 == 0 or done == subset_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{subset_size}  "
                  f"{elapsed:.0f}s  {rate:.0f} rows/s",
                  flush=True)

    insert_time = time.perf_counter() - t0

    # Phase 2: Build the annoy index
    print(f"  Building annoy index (n_trees={params['n_trees']})...")
    t_build = time.perf_counter()
    conn.execute(
        "INSERT INTO vec_items(id, embedding) VALUES ('build-index', NULL)"
    )
    conn.commit()
    build_time = time.perf_counter() - t_build
    print(f"  Build complete in {build_time:.1f}s")

    total_time = time.perf_counter() - t0
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    return {
        "db_path": db_path,
        "insert_time_s": round(insert_time, 3),
        "build_time_s": round(build_time, 3),
        "total_time_s": round(total_time, 3),
        "insert_per_vec_ms": round((insert_time / row_count) * 1000, 2) if row_count else 0,
        "rows": row_count,
        "file_size_mb": round(file_size_mb, 2),
    }


def build_baseline(base_db, ext_path, name, params, subset_size, out_dir):
    btype = params["baseline_type"]
    db_path = os.path.join(out_dir, f"{name}.{subset_size}.db")
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    extra_cols = ""
    if btype == "int8":
        extra_cols = ", embedding_int8 int8[768]"
    elif btype == "bit":
        extra_cols = ", embedding_bq bit[768]"

    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  chunk_size=256,"
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"  {extra_cols})"
    )

    t0 = time.perf_counter()

    insert_sql = {
        "float": "INSERT INTO vec_items(id, embedding) "
                 "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
        "int8":  "INSERT INTO vec_items(id, embedding, embedding_int8) "
                 "SELECT id, vector, vec_quantize_int8(vector, 'unit') "
                 "FROM base.train WHERE id >= :lo AND id < :hi",
        "bit":   "INSERT INTO vec_items(id, embedding, embedding_bq) "
                 "SELECT id, vector, vec_quantize_binary(vector) "
                 "FROM base.train WHERE id >= :lo AND id < :hi",
    }[btype]

    for lo in range(0, subset_size, INSERT_BATCH_SIZE):
        hi = min(lo + INSERT_BATCH_SIZE, subset_size)
        conn.execute(insert_sql, {"lo": lo, "hi": hi})
        conn.commit()
        done = hi
        if done % 5000 == 0 or done == subset_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done:>8}/{subset_size}  "
                  f"{elapsed:.0f}s  {rate:.0f} rows/s",
                  flush=True)

    total_time = time.perf_counter() - t0
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size_mb = os.path.getsize(db_path) / (1024 * 1024)

    return {
        "db_path": db_path,
        "insert_time_s": round(total_time, 3),
        "build_time_s": 0,
        "total_time_s": round(total_time, 3),
        "insert_per_vec_ms": round((total_time / row_count) * 1000, 2) if row_count else 0,
        "rows": row_count,
        "file_size_mb": round(file_size_mb, 2),
    }


def build_index(base_db, ext_path, name, params, subset_size, out_dir):
    if params["index_type"] == "annoy":
        return build_annoy(base_db, ext_path, name, params, subset_size, out_dir)
    else:
        return build_baseline(base_db, ext_path, name, params, subset_size, out_dir)


# ---------- KNN ----------

def measure_knn(db_path, ext_path, base_db, params, subset_size, k=10, n=50):
    """Run KNN queries, measure query time + recall vs brute-force."""
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    query_vectors = load_query_vectors(base_db, n)

    is_baseline = params["index_type"] == "baseline"
    btype = params.get("baseline_type", "")
    oversample = params.get("oversample", 8)

    times_ms = []
    recalls = []
    for qid, query in query_vectors:
        t0 = time.perf_counter()

        if not is_baseline or btype == "float":
            results = conn.execute(
                "SELECT id, distance FROM vec_items "
                "WHERE embedding MATCH :query AND k = :k",
                {"query": query, "k": k},
            ).fetchall()
        elif btype == "int8":
            results = conn.execute(
                "WITH coarse AS ("
                "  SELECT id, embedding FROM vec_items"
                "  WHERE embedding_int8 MATCH vec_quantize_int8(:query, 'unit')"
                "  LIMIT :oversample_k"
                ") "
                "SELECT id, vec_distance_cosine(embedding, :query) as distance "
                "FROM coarse ORDER BY 2 LIMIT :k",
                {"query": query, "k": k, "oversample_k": k * oversample},
            ).fetchall()
        elif btype == "bit":
            results = conn.execute(
                "WITH coarse AS ("
                "  SELECT id, embedding FROM vec_items"
                "  WHERE embedding_bq MATCH vec_quantize_binary(:query)"
                "  LIMIT :oversample_k"
                ") "
                "SELECT id, vec_distance_cosine(embedding, :query) as distance "
                "FROM coarse ORDER BY 2 LIMIT :k",
                {"query": query, "k": k, "oversample_k": k * oversample},
            ).fetchall()

        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)
        result_ids = set(r[0] for r in results)

        # Brute-force ground truth over only the indexed subset
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
        "knn_mean_ms": round(statistics.mean(times_ms), 2),
        "knn_median_ms": round(statistics.median(times_ms), 2),
        "knn_p99_ms": round(sorted(times_ms)[int(len(times_ms) * 0.99)], 2) if len(times_ms) > 1 else round(times_ms[0], 2),
        "recall_at_k": round(statistics.mean(recalls), 4),
        "k": k,
        "n_queries": n,
    }


# ---------- Results ----------

def save_results(out_dir, rows):
    results_path = os.path.join(out_dir, "results.db")
    db = sqlite3.connect(results_path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS index_comparison (
          name             TEXT NOT NULL,
          n_vectors        INTEGER NOT NULL,
          index_type       TEXT NOT NULL,
          n_trees          INTEGER,
          search_k         INTEGER,
          insert_time_s    REAL,
          build_time_s     REAL,
          total_time_s     REAL,
          insert_per_vec_ms REAL,
          file_size_mb     REAL,
          knn_k            INTEGER,
          knn_n_queries    INTEGER,
          knn_mean_ms      REAL,
          knn_median_ms    REAL,
          knn_p99_ms       REAL,
          recall           REAL,
          created_at       TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (name, n_vectors)
        );
    """)
    for r in rows:
        db.execute(
            "INSERT OR REPLACE INTO index_comparison "
            "(name, n_vectors, index_type, n_trees, search_k, "
            " insert_time_s, build_time_s, total_time_s, insert_per_vec_ms, file_size_mb, "
            " knn_k, knn_n_queries, knn_mean_ms, knn_median_ms, knn_p99_ms, recall) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r["name"], r["n_vectors"], r["index_type"],
             r.get("n_trees"), r.get("search_k"),
             r["insert_time_s"], r["build_time_s"], r["total_time_s"],
             r["insert_per_vec_ms"], r["file_size_mb"],
             r["knn_k"], r["knn_n_queries"],
             r["knn_mean_ms"], r["knn_median_ms"], r["knn_p99_ms"],
             r["recall"]),
        )
    db.commit()
    db.close()
    return results_path


def config_description(params):
    if params["index_type"] == "annoy":
        sk = params["search_k"]
        q = params["quantizer"]
        q_str = f" q={q}" if q != "none" else ""
        return (f"annoy  t={params['n_trees']:<3} "
                f"sk={'auto' if sk == 0 else sk}{q_str}")
    else:
        btype = params["baseline_type"]
        if btype in ("int8", "bit"):
            return f"brute  {btype:>6}  (rescore os={params['oversample']})"
        return f"brute  {btype:>6}"


def print_report(all_results):
    print(f"\n{'name':>20} {'N':>7} {'index config':>32}  "
          f"{'ins(s)':>7} {'bld(s)':>7} {'MB':>7} "
          f"{'qry(ms)':>8} {'recall':>7}")
    print("-" * 110)
    for r in all_results:
        bld = f"{r['build_time_s']:.1f}" if r['build_time_s'] > 0 else "-"
        print(f"{r['name']:>20} {r['n_vectors']:>7} {r['config_desc']:>32}  "
              f"{r['insert_time_s']:>7.1f} {bld:>7} {r['file_size_mb']:>7.1f} "
              f"{r['knn_mean_ms']:>8.2f} {r['recall']:>7.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Annoy vs baseline index configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--subset-size", type=int, required=True)
    parser.add_argument("--configs", nargs="+", required=True,
                        help="config specs (name:key=val,...)")
    parser.add_argument("-k", type=int, default=10, help="KNN k (default 10)")
    parser.add_argument("-n", type=int, default=50,
                        help="number of KNN queries for recall (default 50)")
    parser.add_argument("--base-db", default=BASE_DB)
    parser.add_argument("--ext", default=EXT_PATH)
    parser.add_argument("-o", "--out-dir", default=".", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configs = [parse_config(c) for c in args.configs]

    all_results = []
    for i, (name, params) in enumerate(configs, 1):
        desc = config_description(params)
        print(f"\n[{i}/{len(configs)}] {name}  ({desc.strip()})")

        build = build_index(
            args.base_db, args.ext, name, params,
            args.subset_size, args.out_dir,
        )
        print(f"  Insert: {build['insert_time_s']}s  Build: {build['build_time_s']}s  "
              f"Total: {build['total_time_s']}s  {build['file_size_mb']} MB")

        print(f"  Measuring KNN (k={args.k}, n={args.n})...")
        knn = measure_knn(
            build["db_path"], args.ext, args.base_db,
            params, args.subset_size, k=args.k, n=args.n,
        )
        print(f"  KNN: mean={knn['knn_mean_ms']}ms  recall@{args.k}={knn['recall_at_k']}")

        row = {
            "name": name,
            "n_vectors": args.subset_size,
            "index_type": params["index_type"],
            "config_desc": desc,
            "insert_time_s": build["insert_time_s"],
            "build_time_s": build["build_time_s"],
            "total_time_s": build["total_time_s"],
            "insert_per_vec_ms": build["insert_per_vec_ms"],
            "file_size_mb": build["file_size_mb"],
            "knn_k": knn["k"],
            "knn_n_queries": knn["n_queries"],
            "knn_mean_ms": knn["knn_mean_ms"],
            "knn_median_ms": knn["knn_median_ms"],
            "knn_p99_ms": knn["knn_p99_ms"],
            "recall": knn["recall_at_k"],
        }
        if params["index_type"] == "annoy":
            row["n_trees"] = params["n_trees"]
            row["search_k"] = params["search_k"]

        all_results.append(row)

    print_report(all_results)
    results_path = save_results(args.out_dir, all_results)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
