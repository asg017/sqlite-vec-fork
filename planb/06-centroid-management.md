# Step 6: Centroid Management Commands

*This is the key "decoupling" step — it defines how users control training.*

## Overview

Three command verbs handle centroid lifecycle, all via the standard
`INSERT INTO t(t) VALUES (...)` interface:

1. `compute-centroids` — run built-in k-means
2. `set-centroid:N` — import a single externally-computed centroid
3. `assign-vectors` — assign unassigned vectors to nearest centroid

## Command Dispatch

In `vec0Update()`, when the operation is an INSERT and the first column
matches the table name (the "command" pattern used by FTS5 and existing
sqlite-vec `optimize` command):

```c
// In vec0Update(), after existing command checks:
if (strcmp(command, "compute-centroids") == 0 ||
    strncmp(command, "compute-centroids:", 18) == 0) {
  return ivf_cmd_compute_centroids(p, command);
}
if (strncmp(command, "set-centroid:", 13) == 0) {
  // Extract centroid_id from command string, vector from column value
  return ivf_cmd_set_centroid(p, command, vectorData);
}
if (strcmp(command, "assign-vectors") == 0) {
  return ivf_cmd_assign_vectors(p);
}
if (strcmp(command, "clear-centroids") == 0) {
  return ivf_cmd_clear_centroids(p);
}
```

## Command: `compute-centroids`

```sql
-- Basic usage (uses nlist from table definition)
INSERT INTO vec_items(vec_items) VALUES ('compute-centroids');

-- With options (JSON after colon)
INSERT INTO vec_items(vec_items) VALUES ('compute-centroids:{"nlist":256,"max_iterations":20,"seed":42}');
```

### Implementation: `ivf_cmd_compute_centroids()`

```c
static int ivf_cmd_compute_centroids(vec0_vtab *p, const char *command) {
  // 1. Parse optional JSON options
  int nlist = p->vector_columns[0].ivf.nlist;  // default from config
  int max_iter = VEC0_IVF_KMEANS_MAX_ITER;
  uint32_t seed = VEC0_IVF_KMEANS_DEFAULT_SEED;
  // If command has ":{...}", parse JSON for overrides

  // If nlist was 0 at creation and not specified here, error
  if (nlist <= 0) return SQLITE_ERROR;  // "nlist must be specified"

  // 2. Load all vectors into memory
  //    Read from BOTH _ivf_vectors (if recomputing) and _ivf_unassigned
  int N = 0;
  float *allVectors = ivf_load_all_vectors(p, 0, &N);
  if (N < nlist) {
    // Not enough vectors for requested nlist — reduce nlist or error
    nlist = N;  // or: return error "need at least nlist vectors"
  }

  // 3. Run k-means
  int D = p->vector_columns[0].dimensions;
  float *centroids = sqlite3_malloc64(nlist * D * sizeof(float));
  int rc = ivf_kmeans(allVectors, N, D, nlist, max_iter, seed, centroids);

  // 4. Begin savepoint for atomicity
  sqlite3_exec(db, "SAVEPOINT ivf_train", ...);

  // 5. Clear existing centroids
  // DELETE FROM _ivf_centroids
  // DELETE FROM _ivf_vectors  (will re-assign below)

  // 6. Write new centroids
  for (int i = 0; i < nlist; i++) {
    // INSERT INTO _ivf_centroids(centroid_id, centroid) VALUES (?, ?)
  }

  // 7. Assign all vectors to nearest centroid
  //    (re-read from memory, find nearest, insert into _ivf_vectors)
  for (int i = 0; i < N; i++) {
    int cid = ivf_find_nearest_centroid_mem(
      &allVectors[i * D], centroids, D, nlist
    );
    // INSERT INTO _ivf_vectors(rowid, centroid_id, vector) VALUES (?, ?, ?)
  }

  // 8. Clear unassigned table
  // DELETE FROM _ivf_unassigned

  // 9. Update info: ivf_trained = 1, ivf_nlist = nlist
  // UPDATE _info SET value='1' WHERE key='ivf_trained_0'
  // INSERT OR REPLACE INTO _info VALUES('ivf_nlist_0', nlist)

  // 10. Release savepoint
  sqlite3_exec(db, "RELEASE ivf_train", ...);

  // 11. Invalidate cached state
  p->ivfTrained[0] = 1;

  sqlite3_free(centroids);
  sqlite3_free(allVectors);
  return rc;
}
```

## Command: `set-centroid:N`

```sql
-- Import centroid with id=0
INSERT INTO vec_items(vec_items, contents_embedding)
VALUES ('set-centroid:0', :centroid_vector);

-- Import many centroids at once
INSERT INTO vec_items(vec_items, contents_embedding)
SELECT 'set-centroid:' || centroid_idx, centroid_embedding
FROM my_external_centroids;
```

### Implementation: `ivf_cmd_set_centroid()`

```c
static int ivf_cmd_set_centroid(
  vec0_vtab *p,
  const char *command,  // "set-centroid:42"
  const void *vectorData,
  int vectorSize
) {
  // 1. Parse centroid_id from command string
  int centroid_id = atoi(command + 13);  // after "set-centroid:"

  // 2. Validate vector dimensions
  int D = p->vector_columns[0].dimensions;
  if (vectorSize != D * sizeof(float)) return SQLITE_ERROR;

  // 3. Upsert centroid
  // INSERT OR REPLACE INTO _ivf_centroids(centroid_id, centroid) VALUES (?, ?)

  // 4. Update nlist in info (max centroid_id + 1)
  // UPDATE _info SET value = MAX(current, centroid_id+1) WHERE key='ivf_nlist_0'

  // 5. Mark as trained (at least one centroid exists)
  // UPDATE _info SET value='1' WHERE key='ivf_trained_0'
  p->ivfTrained[0] = 1;

  return SQLITE_OK;
}
```

**Design note:** Each `set-centroid` call inserts/replaces one centroid. For
bulk import, the user issues many such inserts (or uses a SELECT...INSERT).
After all centroids are imported, the user calls `assign-vectors` to move
unassigned vectors to their cells.

## Command: `assign-vectors`

```sql
-- Move all unassigned vectors to their nearest centroid cell.
INSERT INTO vec_items(vec_items) VALUES ('assign-vectors');
```

### Implementation: `ivf_cmd_assign_vectors()`

```c
static int ivf_cmd_assign_vectors(vec0_vtab *p) {
  // 1. Check that centroids exist
  if (!ivf_is_trained(p, 0)) {
    return SQLITE_ERROR;  // "no centroids — run compute-centroids first"
  }

  // 2. Load centroids into memory (small — nlist * D * 4 bytes)
  float *centroids = ivf_load_centroids(p, 0, &nlist);

  // 3. Read all unassigned vectors
  // SELECT rowid, vector FROM _ivf_unassigned

  sqlite3_exec(db, "SAVEPOINT ivf_assign", ...);

  while (sqlite3_step(stmt) == SQLITE_ROW) {
    i64 rowid = sqlite3_column_int64(stmt, 0);
    const float *vec = sqlite3_column_blob(stmt, 1);

    // Find nearest centroid
    int cid = ivf_find_nearest_centroid_mem(vec, centroids, D, nlist);

    // INSERT INTO _ivf_vectors(rowid, centroid_id, vector) VALUES (?, ?, ?)
    // DELETE FROM _ivf_unassigned WHERE rowid = ?
  }

  sqlite3_exec(db, "RELEASE ivf_assign", ...);
  sqlite3_free(centroids);
  return SQLITE_OK;
}
```

## Command: `clear-centroids`

```sql
-- Remove all centroids and move all vectors back to unassigned.
INSERT INTO vec_items(vec_items) VALUES ('clear-centroids');
```

This is useful for retraining from scratch or switching to external centroids.

```c
static int ivf_cmd_clear_centroids(vec0_vtab *p) {
  // DELETE FROM _ivf_centroids
  // Move all _ivf_vectors back to _ivf_unassigned:
  //   INSERT INTO _ivf_unassigned SELECT rowid, vector FROM _ivf_vectors
  //   DELETE FROM _ivf_vectors
  // UPDATE _info SET value='0' WHERE key='ivf_trained_0'
  p->ivfTrained[0] = 0;
  return SQLITE_OK;
}
```

## External Computation Workflow Example

```python
import numpy as np
from sklearn.cluster import KMeans
import sqlite3

db = sqlite3.connect("my.db")

# 1. Extract vectors from the unassigned table
rows = db.execute("SELECT rowid, vector FROM __vec_items_ivf_unassigned0").fetchall()
vectors = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])

# 2. Run k-means externally (can be in a thread, on GPU, etc.)
km = KMeans(n_clusters=256, random_state=42)
km.fit(vectors)

# 3. Import centroids back into sqlite-vec
for i, centroid in enumerate(km.cluster_centers_):
    db.execute(
        "INSERT INTO vec_items(vec_items, contents_embedding) VALUES (?, ?)",
        [f"set-centroid:{i}", centroid.tobytes()]
    )

# 4. Assign vectors to cells
db.execute("INSERT INTO vec_items(vec_items) VALUES ('assign-vectors')")
db.commit()
```

## Files Changed

- `sqlite-vec-ivf.c`: Add `ivf_cmd_compute_centroids()`,
  `ivf_cmd_set_centroid()`, `ivf_cmd_assign_vectors()`,
  `ivf_cmd_clear_centroids()`, `ivf_load_all_vectors()`,
  `ivf_load_centroids()`.
- `sqlite-vec.c`: ~8-line command dispatch block in `vec0Update()` that
  checks for `"compute-centroids"`, `"set-centroid:"`, `"assign-vectors"`,
  `"clear-centroids"` prefixes and calls into `sqlite-vec-ivf.c` handlers.

## Python Tests

```python
def test_compute_centroids(db):
    db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[4] indexed by ivf(nlist=4))")
    # Insert 100 vectors
    for i in range(100):
        db.execute("INSERT INTO t(rowid, v) VALUES (?, ?)",
                   [i, _f32([i % 10, i // 10, 0, 0])])

    # Before training: all unassigned
    assert db.execute("SELECT count(*) FROM __t_ivf_unassigned0").fetchone()[0] == 100

    # Train
    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")

    # After training: all assigned, none unassigned
    assert db.execute("SELECT count(*) FROM __t_ivf_unassigned0").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM __t_ivf_vectors0").fetchone()[0] == 100
    assert db.execute("SELECT count(*) FROM __t_ivf_centroids0").fetchone()[0] == 4

def test_set_centroid_manual(db):
    db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[4] indexed by ivf(nlist=0))")
    for i in range(20):
        db.execute("INSERT INTO t(rowid, v) VALUES (?, ?)", [i, _f32([i,0,0,0])])

    # Manually set 2 centroids
    db.execute("INSERT INTO t(t, v) VALUES ('set-centroid:0', ?)", [_f32([5,0,0,0])])
    db.execute("INSERT INTO t(t, v) VALUES ('set-centroid:1', ?)", [_f32([15,0,0,0])])

    # Assign
    db.execute("INSERT INTO t(t) VALUES ('assign-vectors')")

    assert db.execute("SELECT count(*) FROM __t_ivf_unassigned0").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM __t_ivf_vectors0").fetchone()[0] == 20

def test_clear_centroids(db):
    # ... setup and train ...
    db.execute("INSERT INTO t(t) VALUES ('clear-centroids')")
    assert db.execute("SELECT count(*) FROM __t_ivf_centroids0").fetchone()[0] == 0
    assert db.execute("SELECT count(*) FROM __t_ivf_unassigned0").fetchone()[0] == 100
```
