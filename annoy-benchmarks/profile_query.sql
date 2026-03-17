-- Annoy KNN query profiling workload
-- Builds a 10k-vector annoy index, then runs 100 KNN queries.
-- Use with: perf record -g dist/sqlite3-profile < profile_query.sql
.timer on

-- Phase 1: Create annoy-indexed table
ATTACH DATABASE 'annoy-benchmarks/seed/base.db' AS base;

CREATE VIRTUAL TABLE vec_items USING vec0(
  id integer primary key,
  embedding float[768] distance_metric=cosine
    INDEXED BY annoy(n_trees=25)
);

-- Phase 2: Insert 10k vectors
.print "--- INSERT 10000 vectors ---"
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id < 10000;

-- Phase 3: Build annoy index
.print "--- BUILD INDEX ---"
INSERT INTO vec_items(id, embedding) VALUES ('build-index', NULL);

-- Phase 4: Run 100 KNN queries (this is the hot path we want to profile)
.print "--- KNN QUERIES (100x k=10) ---"

-- Create a temp table of queries to iterate
CREATE TEMP TABLE queries AS
  SELECT id, vector FROM base.query_vectors ORDER BY id LIMIT 100;

-- Run queries one by one via a CTE trick: select each query row and MATCH
-- We use a loop via recursive CTE to avoid Python
SELECT count(*) FROM (
  SELECT v.id, v.distance
  FROM queries q, vec_items v
  WHERE v.embedding MATCH q.vector AND v.k = 10
);

.print "--- DONE ---"
