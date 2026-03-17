#!/usr/bin/env python3
"""Compute per-subset ground truth for DiskANN benchmarks.

For subset sizes < 1M, the full-dataset ground truth is invalid because many
true neighbors won't be in the subset. This script builds a temporary vec0
float table with the first N vectors and runs brute-force KNN to get correct
ground truth per subset.

For 1M (the full dataset), it converts the existing `neighbors` table.
"""
import argparse
import os
import sqlite3
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXT_PATH = os.path.join(_SCRIPT_DIR, "..", "dist", "vec0")
BASE_DB = os.path.join(_SCRIPT_DIR, "seed", "base.db")
FULL_DATASET_SIZE = 1_000_000


def gen_ground_truth_subset(base_db, ext_path, subset_size, n_queries, k, out_path):
    """Build ground truth by brute-force KNN over the first `subset_size` vectors."""
    if os.path.exists(out_path):
        os.remove(out_path)

    conn = sqlite3.connect(out_path)
    conn.enable_load_extension(True)
    conn.load_extension(ext_path)

    conn.execute(
        "CREATE TABLE ground_truth ("
        "  query_vector_id INTEGER NOT NULL,"
        "  rank INTEGER NOT NULL,"
        "  neighbor_id INTEGER NOT NULL,"
        "  distance REAL NOT NULL,"
        "  PRIMARY KEY (query_vector_id, rank)"
        ")"
    )

    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    # Build a temp vec0 table with the subset
    print(f"  Building temp vec0 table with {subset_size} vectors...")
    conn.execute(
        "CREATE VIRTUAL TABLE tmp_vec USING vec0("
        "  id integer primary key,"
        "  embedding float[768] distance_metric=cosine"
        ")"
    )

    t0 = time.perf_counter()
    conn.execute(
        "INSERT INTO tmp_vec(id, embedding) "
        "SELECT id, vector FROM base.train WHERE id < :n",
        {"n": subset_size},
    )
    conn.commit()
    build_time = time.perf_counter() - t0
    print(f"  Temp table built in {build_time:.1f}s")

    # Load query vectors
    query_vectors = conn.execute(
        "SELECT id, vector FROM base.query_vectors ORDER BY id LIMIT :n",
        {"n": n_queries},
    ).fetchall()

    print(f"  Running brute-force KNN for {len(query_vectors)} queries, k={k}...")
    t0 = time.perf_counter()

    for i, (qid, qvec) in enumerate(query_vectors):
        results = conn.execute(
            "SELECT id, distance FROM tmp_vec "
            "WHERE embedding MATCH :query AND k = :k",
            {"query": qvec, "k": k},
        ).fetchall()

        for rank, (nid, dist) in enumerate(results):
            conn.execute(
                "INSERT INTO ground_truth(query_vector_id, rank, neighbor_id, distance) "
                "VALUES (?, ?, ?, ?)",
                (qid, rank, nid, dist),
            )

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / (i + 1)) * (len(query_vectors) - i - 1)
            print(f"    {i+1}/{len(query_vectors)} queries  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                  flush=True)

    conn.commit()

    # Drop temp table
    conn.execute("DROP TABLE tmp_vec")
    conn.execute("DETACH DATABASE base")
    conn.commit()

    elapsed = time.perf_counter() - t0
    total_rows = conn.execute("SELECT count(*) FROM ground_truth").fetchone()[0]
    conn.close()

    print(f"  Ground truth: {total_rows} rows in {elapsed:.1f}s -> {out_path}")


def gen_ground_truth_full(base_db, n_queries, k, out_path):
    """Convert the existing neighbors table for the full 1M dataset."""
    if os.path.exists(out_path):
        os.remove(out_path)

    conn = sqlite3.connect(out_path)
    conn.execute(f"ATTACH DATABASE '{base_db}' AS base")

    conn.execute(
        "CREATE TABLE ground_truth ("
        "  query_vector_id INTEGER NOT NULL,"
        "  rank INTEGER NOT NULL,"
        "  neighbor_id INTEGER NOT NULL,"
        "  distance REAL,"
        "  PRIMARY KEY (query_vector_id, rank)"
        ")"
    )

    # neighbors_id is TEXT in the original table, cast to INT
    conn.execute(
        "INSERT INTO ground_truth(query_vector_id, rank, neighbor_id) "
        "SELECT query_vector_id, rank, CAST(neighbors_id AS INTEGER) "
        "FROM base.neighbors "
        "WHERE query_vector_id < :n AND rank < :k",
        {"n": n_queries, "k": k},
    )
    conn.commit()

    total_rows = conn.execute("SELECT count(*) FROM ground_truth").fetchone()[0]
    conn.execute("DETACH DATABASE base")
    conn.close()
    print(f"  Ground truth (full): {total_rows} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate per-subset ground truth")
    parser.add_argument("--subset-size", type=int, required=True, help="number of vectors in subset")
    parser.add_argument("-n", type=int, default=100, help="number of query vectors")
    parser.add_argument("-k", type=int, default=100, help="max k for ground truth")
    parser.add_argument("--base-db", default=BASE_DB)
    parser.add_argument("--ext", default=EXT_PATH)
    parser.add_argument("-o", "--out-dir", default=".", help="output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"ground_truth.{args.subset_size}.db")

    if args.subset_size >= FULL_DATASET_SIZE:
        gen_ground_truth_full(args.base_db, args.n, args.k, out_path)
    else:
        gen_ground_truth_subset(
            args.base_db, args.ext, args.subset_size, args.n, args.k, out_path
        )


if __name__ == "__main__":
    main()
