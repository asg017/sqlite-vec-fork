-- Profile ONLY the KNN query path (build done separately)
-- Usage:
--   1. First run profile_query.sql to create :memory: DB (or use pre-built)
--   2. This script attaches a pre-built annoy DB and runs queries only
.timer on

ATTACH DATABASE 'annoy-benchmarks/seed/base.db' AS base;

-- Use the pre-built DB from bench.py run
ATTACH DATABASE 'annoy-benchmarks/runs/profile/annoy-profile.10000.db' AS idx;

.print "--- KNN QUERIES (200x k=10) ---"

SELECT count(*) FROM (
  SELECT v.id, v.distance
  FROM (SELECT id, vector FROM base.query_vectors ORDER BY id LIMIT 200) q,
       idx.vec_items v
  WHERE v.embedding MATCH q.vector AND v.k = 10
);

.print "--- DONE ---"
