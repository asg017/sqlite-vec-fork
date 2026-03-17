#!/usr/bin/env python3
"""Run many KNN queries for CPU profiling. Used by: make profile-sample-query"""
import os
import sqlite3
import sys

DB_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ivf_profile_query.db"
BASE_DB = os.path.join(
    os.path.dirname(__file__), "..", "benchmark2", "zilliz", "seed", "base.db"
)
N_QUERIES = 200
K = 10

conn = sqlite3.connect(DB_PATH)
conn.enable_load_extension(True)
conn.load_extension(os.path.join(os.path.dirname(__file__), "..", "dist", "vec0"))
conn.execute(f"ATTACH DATABASE '{BASE_DB}' AS base")

# Load query vectors
queries = conn.execute(
    "SELECT id, vector FROM base.query_vectors ORDER BY id LIMIT ?", (N_QUERIES,)
).fetchall()

for qid, qvec in queries:
    conn.execute(
        "SELECT id, distance FROM vec_items WHERE embedding MATCH ? AND k = ?",
        (qvec, K),
    ).fetchall()

conn.close()
