-- Profile IVF inserts at scale: 50k vectors into a trained index with nlist=128
-- Run: make cli && dist/sqlite3 < ivf-benchmarks/profile_insert_1m.sql

.timer on

ATTACH DATABASE 'benchmark2/zilliz/seed/base.db' AS base;
PRAGMA page_size=8192;

CREATE VIRTUAL TABLE vec_items USING vec0(
  id integer primary key,
  embedding float[768] distance_metric=cosine
    indexed by ivf(nlist=128, nprobe=16)
);

-- Insert training vectors
.print "=== Insert 8192 training vectors ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id < 8192;

-- Train
.print "=== Train k-means ==="
INSERT INTO vec_items(id) VALUES ('compute-centroids');

-- Insert in batches to see scaling
.print "=== Insert 10k trained (8k-18k) ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 8192 AND id < 18192;

.print "=== Insert 10k trained (18k-28k) ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 18192 AND id < 28192;

.print "=== Insert 10k trained (28k-38k) ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 28192 AND id < 38192;

.print "=== Insert 10k trained (38k-48k) ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 38192 AND id < 48192;

.print "=== Insert 10k trained (48k-58k) ==="
INSERT INTO vec_items(id, embedding)
  SELECT id, vector FROM base.train WHERE id >= 48192 AND id < 58192;
