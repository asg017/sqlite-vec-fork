# Step 4: IVF Insert Path

*Reference: `plans/diskann-plans/08-insert.md` for the DiskANN equivalent.*

## Overview

The insert path is simple compared to DiskANN — no graph maintenance. The
behavior depends on whether centroids have been computed.

## Insert Logic: `ivf_insert()`

```c
static int ivf_insert(
  vec0_vtab *p,
  int vectorColumnIdx,
  i64 rowid,
  const void *vectorData,
  int vectorSize
) {
  struct Vec0IvfConfig *cfg = &p->vector_columns[vectorColumnIdx].ivf;

  if (!ivf_is_trained(p, vectorColumnIdx)) {
    // Flat mode: store in unassigned table
    return ivf_insert_unassigned(p, vectorColumnIdx, rowid, vectorData, vectorSize);
  } else {
    // Trained mode: find nearest centroid, store in vectors table
    int centroid_id;
    int rc = ivf_find_nearest_centroid(p, vectorColumnIdx, vectorData, vectorSize, &centroid_id);
    if (rc != SQLITE_OK) return rc;
    return ivf_insert_assigned(p, vectorColumnIdx, rowid, centroid_id, vectorData, vectorSize);
  }
}
```

### `ivf_is_trained()`

Check the `_info` table for `ivf_trained_{NN}` = '1'.

Cache this value in `vec0_vtab` to avoid repeated lookups:

```c
struct vec0_vtab {
  // ...
  int ivfTrained[VEC0_MAX_VECTOR_COLUMNS]; // cached: -1=unknown, 0=no, 1=yes
};
```

### `ivf_insert_unassigned()`

```c
// INSERT INTO _ivf_unassigned(rowid, vector) VALUES (?, ?)
sqlite3_bind_int64(stmt, 1, rowid);
sqlite3_bind_blob(stmt, 2, vectorData, vectorSize, SQLITE_STATIC);
sqlite3_step(stmt);
```

### `ivf_find_nearest_centroid()`

Scan all centroids and return the one with minimum distance:

```c
static int ivf_find_nearest_centroid(
  vec0_vtab *p,
  int colIdx,
  const void *query,
  int querySize,
  int *out_centroid_id
) {
  // SELECT centroid_id, centroid FROM _ivf_centroids
  float min_dist = FLT_MAX;
  int best_id = -1;

  while (sqlite3_step(stmtReadAll) == SQLITE_ROW) {
    int cid = sqlite3_column_int(stmtReadAll, 0);
    const void *centroid = sqlite3_column_blob(stmtReadAll, 1);
    float dist = compute_distance(query, centroid, dims,
      p->vector_columns[colIdx].distance_metric);
    if (dist < min_dist) {
      min_dist = dist;
      best_id = cid;
    }
  }

  *out_centroid_id = best_id;
  return SQLITE_OK;
}
```

**Performance note:** With nlist up to a few thousand, scanning all centroids
is fast (a few thousand distance computations on small vectors). For very large
nlist, we could cache centroids in memory, but that's an optimization for later.

### `ivf_insert_assigned()`

```c
// INSERT INTO _ivf_vectors(rowid, centroid_id, vector) VALUES (?, ?, ?)
sqlite3_bind_int64(stmt, 1, rowid);
sqlite3_bind_int(stmt, 2, centroid_id);
sqlite3_bind_blob(stmt, 3, vectorData, vectorSize, SQLITE_STATIC);
sqlite3_step(stmt);
```

## Integration Point

In `vec0Update_Insert()`, same pattern as DiskANN (line ~10736):

```c
for (int i = 0; i < p->numVectorColumns; i++) {
  if (p->vector_columns[i].ivf.enabled) {
    rc = ivf_insert(p, i, rowid, vectorDatas[i], vectorSizes[i]);
    if (rc != SQLITE_OK) goto done;
  }
}
```

## Delete Path: `ivf_delete()`

On DELETE, remove the vector from whichever table it's in:

```c
static int ivf_delete(vec0_vtab *p, int colIdx, i64 rowid) {
  // Try both tables — vector is in one or the other
  // DELETE FROM _ivf_vectors WHERE rowid = ?
  // DELETE FROM _ivf_unassigned WHERE rowid = ?
  return SQLITE_OK;
}
```

No graph repair needed (unlike DiskANN step 9). This is a major simplicity win.

## Update Path

For UPDATE on a vector column:
1. Delete old entry (from vectors or unassigned).
2. Insert new entry (follows same trained/untrained logic).

## Files Changed

- `sqlite-vec-ivf.c`: Add `ivf_insert()`, `ivf_delete()`,
  `ivf_insert_unassigned()`, `ivf_insert_assigned()`,
  `ivf_find_nearest_centroid()`, `ivf_is_trained()`.
- `sqlite-vec.c`: ~5-line loop in `vec0Update_Insert()` calling
  `ivf_insert()` (same pattern as DiskANN dispatch at line ~10736).
  ~3-line call to `ivf_delete()` in `vec0Update_Delete()`.

## Python Tests

```python
def test_ivf_insert_flat_mode(db):
    """Before training, vectors go to unassigned table."""
    db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[4] indexed by ivf(nlist=4))")
    db.execute("INSERT INTO t(rowid, v) VALUES (1, ?)", [_f32([1,2,3,4])])

    # Should be in unassigned
    assert db.execute("SELECT count(*) FROM __t_ivf_unassigned0").fetchone()[0] == 1
    assert db.execute("SELECT count(*) FROM __t_ivf_vectors0").fetchone()[0] == 0

def test_ivf_insert_trained_mode(db):
    """After training, vectors go to assigned table."""
    db.execute("CREATE VIRTUAL TABLE t USING vec0(v float[4] indexed by ivf(nlist=2))")
    for i in range(20):
        db.execute("INSERT INTO t(rowid, v) VALUES (?, ?)", [i, _f32([i,i,i,i])])

    db.execute("INSERT INTO t(t) VALUES ('compute-centroids')")

    # Insert new vector — should go to assigned table
    db.execute("INSERT INTO t(rowid, v) VALUES (100, ?)", [_f32([5,5,5,5])])
    row = db.execute("SELECT centroid_id FROM __t_ivf_vectors0 WHERE rowid=100").fetchone()
    assert row is not None
    assert row[0] in (0, 1)  # assigned to one of the 2 centroids
```
