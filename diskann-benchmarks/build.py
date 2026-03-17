#!/usr/bin/env python3
"""Build DiskANN and baseline indexes for benchmarking."""
import argparse
import os
import sqlite3
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_PATH = os.path.join(_SCRIPT_DIR, "..", "dist", "vec0")
BASE_DB = os.path.join(_SCRIPT_DIR, "seed", "base.db")

DISKANN_CONFIGS = {
    "diskann-binary": {"quantizer": "binary", "R": 72, "L": 128},
    "diskann-binary-R24": {"quantizer": "binary", "R": 24, "L": 128},
    "diskann-binary-R48": {"quantizer": "binary", "R": 48, "L": 128},
    "diskann-binary-L64": {"quantizer": "binary", "R": 72, "L": 64},
    "diskann-binary-L256": {"quantizer": "binary", "R": 72, "L": 256},
    "diskann-int8": {"quantizer": "int8", "R": 72, "L": 128},
}

BASELINE_CONFIGS = {
    "baseline-float": {"mode": "float"},
    "baseline-bq": {"mode": "bq"},
    "baseline-int8": {"mode": "int8"},
}

ALL_CONFIGS = list(DISKANN_CONFIGS.keys()) + list(BASELINE_CONFIGS.keys())
BATCH_SIZE = 1000


def db_path_for(out_dir, config_name, subset_size):
    return os.path.join(out_dir, f"{config_name}.{subset_size}.db")


def build_diskann(base_db, ext_path, config_name, subset_size, out_dir):
    """Build a DiskANN index."""
    cfg = DISKANN_CONFIGS[config_name]
    db_path = db_path_for(out_dir, config_name, subset_size)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"    INDEXED BY diskann("
        f"      neighbor_quantizer={cfg['quantizer']},"
        f"      n_neighbors={cfg['R']},"
        f"      search_list_size={cfg['L']}"
        f"    )"
        f")"
    )

    t0 = time.perf_counter()

    # Batched inserts with progress reporting
    for lo in range(0, subset_size, BATCH_SIZE):
        hi = min(lo + BATCH_SIZE, subset_size)
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id >= :lo AND id < :hi",
            {"lo": lo, "hi": hi},
        )
        conn.commit()

        done = hi
        if done % 1000 == 0 or done == subset_size:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (subset_size - done) / rate if rate > 0 else 0
            print(
                f"    {done:>8}/{subset_size}  "
                f"{elapsed:.0f}s elapsed  "
                f"{rate:.0f} rows/s  "
                f"eta {eta:.0f}s",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size = os.path.getsize(db_path)

    return {
        "config_name": config_name,
        "subset_size": subset_size,
        "db_path": db_path,
        "build_time_s": round(elapsed, 3),
        "rows": row_count,
        "file_size_mb": round(file_size / (1024 * 1024), 1),
    }


def build_baseline(base_db, ext_path, config_name, subset_size, out_dir):
    """Build a baseline (chunk-based) index."""
    cfg = BASELINE_CONFIGS[config_name]
    mode = cfg["mode"]
    db_path = db_path_for(out_dir, config_name, subset_size)

    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)
    conn.execute("PRAGMA page_size=8192")
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    extra_cols = ""
    if mode == "bq":
        extra_cols = ", embedding_bq bit[768]"
    elif mode == "int8":
        extra_cols = ", embedding_int8 int8[768]"

    conn.execute(
        f"CREATE VIRTUAL TABLE vec_items USING vec0("
        f"  chunk_size=256,"
        f"  id integer primary key,"
        f"  embedding float[768] distance_metric=cosine"
        f"{extra_cols})"
    )

    t0 = time.perf_counter()

    if mode == "bq":
        conn.execute(
            "INSERT INTO vec_items(id, embedding, embedding_bq) "
            "SELECT id, vector, vec_quantize_binary(vector) FROM base.train "
            "WHERE id < :n",
            {"n": subset_size},
        )
    elif mode == "int8":
        conn.execute(
            "INSERT INTO vec_items(id, embedding, embedding_int8) "
            "SELECT id, vector, vec_quantize_int8(vector, 'unit') FROM base.train "
            "WHERE id < :n",
            {"n": subset_size},
        )
    else:
        conn.execute(
            "INSERT INTO vec_items(id, embedding) "
            "SELECT id, vector FROM base.train WHERE id < :n",
            {"n": subset_size},
        )

    conn.commit()
    elapsed = time.perf_counter() - t0

    row_count = conn.execute("SELECT count(*) FROM vec_items").fetchone()[0]
    conn.close()
    file_size = os.path.getsize(db_path)

    return {
        "config_name": config_name,
        "subset_size": subset_size,
        "db_path": db_path,
        "build_time_s": round(elapsed, 3),
        "rows": row_count,
        "file_size_mb": round(file_size / (1024 * 1024), 1),
    }


def build_config(base_db, ext_path, config_name, subset_size, out_dir):
    if config_name in DISKANN_CONFIGS:
        return build_diskann(base_db, ext_path, config_name, subset_size, out_dir)
    else:
        return build_baseline(base_db, ext_path, config_name, subset_size, out_dir)


def save_results(results, out_dir):
    results_path = os.path.join(out_dir, "results.db")
    db = sqlite3.connect(results_path)
    schema_path = os.path.join(os.path.dirname(__file__), "SCHEMA.sql")
    with open(schema_path) as f:
        db.executescript(f.read())
    for r in results:
        db.execute(
            "INSERT OR REPLACE INTO build_results "
            "(config_name, subset_size, db_path, build_time_s, rows, file_size_mb) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r["config_name"], r["subset_size"], r["db_path"],
             r["build_time_s"], r["rows"], r["file_size_mb"]),
        )
    db.commit()
    db.close()
    return results_path


def main():
    parser = argparse.ArgumentParser(description="Build DiskANN and baseline indexes")
    parser.add_argument("--subset-size", type=int, required=True, help="number of vectors")
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS,
                        choices=ALL_CONFIGS, help="configs to build")
    parser.add_argument("--base-db", default=BASE_DB)
    parser.add_argument("--ext", default=EXT_PATH)
    parser.add_argument("-o", "--out-dir", default=".", help="output directory")
    parser.add_argument("--skip-if-exists", action="store_true",
                        help="skip building if db file already exists")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = []
    total = len(args.configs)

    for i, config_name in enumerate(args.configs, 1):
        db_path = db_path_for(args.out_dir, config_name, args.subset_size)
        if args.skip_if_exists and os.path.exists(db_path):
            print(f"[{i}/{total}] Skipping {config_name} (exists: {db_path})")
            continue

        print(f"[{i}/{total}] Building {config_name} (n={args.subset_size})...", flush=True)
        result = build_config(args.base_db, args.ext, config_name, args.subset_size,
                              args.out_dir)
        results.append(result)
        print(f"  -> {result['build_time_s']}s, {result['rows']} rows, "
              f"{result['file_size_mb']} MB")

    if results:
        print()
        print(f"{'config':>25} {'subset':>8} {'time (s)':>10} {'size (MB)':>10}")
        print("-" * 58)
        for r in results:
            print(f"{r['config_name']:>25} {r['subset_size']:>8} "
                  f"{r['build_time_s']:>10.3f} {r['file_size_mb']:>10.1f}")

        results_path = save_results(results, args.out_dir)
        print(f"\nBuild results saved to {results_path}")


if __name__ == "__main__":
    main()
