-- IVF profiling script. Run with:
--   dist/sqlite3-vec-profile < ivf-benchmarks/profile_ivf.sql
-- Then profile with:
--   sample dist/sqlite3-vec-profile 5 -file profile.txt
-- Or Instruments: Product > Profile > Time Profiler

.timer on

ATTACH DATABASE 'benchmark2/zilliz/seed/base.db' AS base;

PRAGMA page_size=8192;

-- Create IVF table
CREATE VIRTUAL TABLE vec_items USING vec0(
  id integer primary key,
  embedding float[768] distance_metric=cosine
    indexed by ivf(nlist=16, nprobe=4)
);

-- Phase 1: Insert 316 vectors (sqrt(100000))
.print "=== Phase 1: Insert 316 vectors (training sample) ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id < 316;

-- Phase 2: Train k-means
.print "=== Phase 2: Train k-means (nlist=16 on 316 vectors) ==="
INSERT INTO vec_items(id) VALUES ('compute-centroids');

-- Phase 3: Insert remaining vectors (auto-assigned)
-- Do in batches to see per-batch timing
.print "=== Phase 3: Insert 9684 vectors (auto-assigned) ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 316 AND id < 10000;

.print "=== Phase 3b: Insert next 10000 ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 10000 AND id < 20000;

.print "=== Phase 3c: Insert next 10000 ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 20000 AND id < 30000;

.print "=== Queries ==="
-- Run 10 KNN queries
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=0) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=1) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=2) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=3) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=4) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=5) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=6) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=7) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=8) AND k=10;
SELECT id, distance FROM vec_items WHERE embedding MATCH (SELECT vector FROM base.query_vectors WHERE id=9) AND k=10;
