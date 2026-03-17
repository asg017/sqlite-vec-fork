.timer on
ATTACH DATABASE 'annoy-benchmarks/seed/base.db' AS base;
ATTACH DATABASE 'annoy-benchmarks/runs/profile/annoy-profile.50000.db' AS idx;

.print "--- KNN QUERIES (50x k=10) on 50k vectors ---"
SELECT count(*) FROM (
  SELECT v.id, v.distance
  FROM (SELECT id, vector FROM base.query_vectors ORDER BY id LIMIT 50) q,
       idx.vec_items v
  WHERE v.embedding MATCH q.vector AND v.k = 10
);
.print "--- DONE ---"
