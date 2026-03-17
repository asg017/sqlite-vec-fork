# IVF Index Proposal for sqlite-vec

## Motivation

DiskANN graph-based indexing showed poor insert performance and modest speedups
for sqlite-vec's use case (disk-native, single-file, transaction-safe). IVF
(Inverted File Index) is a simpler approach that trades a small amount of recall
for large speedups, with much cheaper inserts and a more natural fit for
SQLite's row-oriented storage.

## Design Philosophy

1. **Decouple everything.** Table creation, vector insertion, centroid
   computation, and KNN queries are independent operations. The user controls
   when (and how) centroids are computed.

2. **Graceful degradation.** Before centroids exist, the index operates in
   "flat mode" — brute-force scan over all vectors. This means the table is
   always queryable, even before training.

3. **User-controllable training.** Two paths to centroids:
   - **Automatic**: `INSERT INTO t(t) VALUES ('compute-centroids')` runs
     built-in k-means.
   - **Manual**: User computes centroids externally and inserts them via a
     structured command, enabling custom algorithms, different threads, or
     even different machines.

4. **Incremental updates.** After initial training, new inserts are assigned to
   the nearest existing centroid. Centroids can be recomputed at any time to
   account for distribution drift.

5. **Disk-native.** All state lives in shadow tables. No in-memory graph.
   Centroid data is small enough to cache in prepared statements.

## SQL API

### Table Creation

```sql
CREATE VIRTUAL TABLE vec_items USING vec0(
  id INTEGER PRIMARY KEY,
  contents_embedding float[1024] distance_metric=cosine INDEXED BY ivf(
    nlist=256,
    nprobe=16
  )
);
```

**IVF parameters:**
- `nlist` — number of Voronoi cells / centroids (default: 128). Can be 0 to
  defer the decision until `compute-centroids` time.
- `nprobe` — number of cells to search at query time (default: 10). Higher =
  better recall, slower queries.

The distance metric is specified via the existing `distance_metric=` option on
the vector column itself (not inside `ivf(...)`). This is already supported by
vec0 for `l2`, `cosine`, and `ip`. IVF reuses the column's `distance_metric`
for both centroid assignment and KNN distance computation.

### Inserting Vectors

```sql
-- Works immediately, even before centroids exist.
INSERT INTO vec_items(id, contents_embedding)
VALUES (1, :vector);
```

- **Before centroids**: vector stored in `_ivf_unassigned{NN}` shadow table.
- **After centroids**: vector stored in `_ivf_vectors{NN}`, with a
  `centroid_id` assignment computed on insert.

### Computing Centroids (Automatic)

```sql
-- Runs k-means on all vectors, populates centroid shadow table.
-- This is a blocking operation that locks the table.
INSERT INTO vec_items(vec_items)
VALUES ('compute-centroids');
```

Optional parameters via JSON:

```sql
INSERT INTO vec_items(vec_items)
VALUES ('compute-centroids:{"nlist":256,"max_iterations":20,"seed":42}');
```

This:
1. Reads all vectors (from both `_ivf_vectors` and `_ivf_unassigned`).
2. Runs k-means with the configured `nlist`.
3. Writes centroids to `_ivf_centroids{NN}`.
4. Assigns (or re-assigns) every vector to its nearest centroid.
5. Moves any unassigned vectors into `_ivf_vectors{NN}`.

### Computing Centroids (Manual / External)

Users can compute centroids externally (e.g., using faiss, scikit-learn, a
separate thread) and import them:

```sql
-- Clear existing centroids and import new ones.
-- centroid_id is a 0-based integer, centroid_vector is the embedding.
INSERT INTO vec_items(vec_items, contents_embedding)
VALUES ('set-centroids:0', :centroid_0),
       ('set-centroids:1', :centroid_1),
       ...
       ('set-centroids:255', :centroid_255);
```

Or in a single batch command:

```sql
-- Import all centroids from a result set.
INSERT INTO vec_items(vec_items, contents_embedding)
SELECT 'set-centroid:' || idx, centroid_vector
FROM my_external_centroids;
```

After importing centroids, the user must trigger assignment:

```sql
-- Assign all unassigned vectors to their nearest centroid.
INSERT INTO vec_items(vec_items)
VALUES ('assign-vectors');
```

### Recomputing / Updating Centroids

```sql
-- Recompute centroids from current vector distribution.
-- Re-assigns all vectors afterward.
INSERT INTO vec_items(vec_items)
VALUES ('compute-centroids');
```

This is idempotent — it replaces existing centroids entirely.

To incrementally add vectors without recomputing:

```sql
-- New inserts after training auto-assign to nearest centroid.
INSERT INTO vec_items(id, contents_embedding) VALUES (1001, :vec);
```

### Querying (KNN)

```sql
-- Same API as existing vec0 KNN queries.
SELECT id, distance
FROM vec_items
WHERE contents_embedding MATCH :query
  AND k = 20;
```

**Behavior depends on state:**
- **No centroids**: full brute-force scan (flat mode).
- **Centroids exist, unassigned vectors remain**: IVF probe of `nprobe`
  cells + brute-force scan of unassigned vectors, results merged.
- **Centroids exist, all assigned**: pure IVF probe of `nprobe` cells.

### Inspecting State

```sql
-- Check if centroids have been computed.
SELECT value FROM vec_items_info WHERE key = 'ivf_trained_0';
-- Returns '1' if centroids exist, '0' otherwise.

-- Count unassigned vectors.
SELECT count(*) FROM vec_items_ivf_unassigned0;

-- Read centroids directly (shadow table access).
SELECT rowid, centroid FROM vec_items_ivf_centroids0;
```

## File Structure

All IVF code lives in separate files, `#include`d into `sqlite-vec.c` to keep
the diff minimal. The compiler sees one translation unit — no build changes.

```
sqlite-vec-ivf-kmeans.c   — Pure k-means algorithm (no SQLite dependency)
                             ivf_kmeans(), ivf_kmeans_init_plusplus()

sqlite-vec-ivf.c          — All IVF SQLite integration:
                             structs, constants, parser, shadow tables,
                             insert/delete, centroid commands, KNN query

sqlite-vec.c              — Minimal additions only:
                             #include "sqlite-vec-ivf-kmeans.c"
                             #include "sqlite-vec-ivf.c"
                             + struct Vec0IvfConfig ivf; in VectorColumnDefinition
                             + ivf fields in vec0_vtab
                             + dispatch calls (5-10 lines each) in:
                               vec0_parse_vector_column()
                               vec0Create()
                               vec0Update_Insert()
                               vec0Update_Delete()
                               vec0Filter_knn()
                               vec0BestIndex()
```

**Goal:** `sqlite-vec.c` changes should be under ~50 lines total — just struct
fields and one-line dispatch calls into functions defined in `sqlite-vec-ivf.c`.

## Comparison with DiskANN

| Aspect | DiskANN | IVF |
|--------|---------|-----|
| Insert cost | O(L * R) graph ops per vector | O(nlist) distance comparisons (with centroids) or O(1) (unassigned) |
| Query cost | O(L * R * log L) beam search | O(nprobe * n/nlist) linear scan per cell |
| Memory | Graph nodes + quantized vectors | Centroids (small) + cell assignments |
| Training required | No (online) | Yes, but deferred and optional |
| Recall control | search_list_size, alpha | nprobe (easy linear tradeoff) |
| Complexity | High (graph maintenance) | Low (clustering + assignment) |
