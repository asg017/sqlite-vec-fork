-- Annoy KNN query profiling workload (50k vectors)
-- Larger dataset to make query bottlenecks more visible.
.timer on

ATTACH DATABASE 'annoy-benchmarks/seed/base.db' AS base;

CREATE VIRTUAL TABLE vec_items USING vec0(
  id integer primary key,
  embedding float[768] distance_metric=cosine
    INDEXED BY annoy(n_trees=25)
);

.print "--- INSERT 50000 vectors ---"
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id < 50000;

.print "--- BUILD INDEX ---"
INSERT INTO vec_items(id, embedding) VALUES ('build-index', NULL);

.print "--- KNN QUERIES (50x k=10) ---"
SELECT count(*) FROM (
  SELECT v.id, v.distance
  FROM (SELECT id, vector FROM base.query_vectors ORDER BY id LIMIT 50) q,
       vec_items v
  WHERE v.embedding MATCH q.vector AND v.k = 10
);

.print "--- DONE ---"
