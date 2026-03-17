# Step 3: IVF Shadow Tables

*Reference: `plans/diskann-plans/03-shadow-tables.md` for the DiskANN equivalent.*

## Overview

Create three shadow tables per IVF-indexed vector column. The key insight is
that centroids are stored separately and can be NULL/empty initially, enabling
the "flat mode then train" workflow.

## Shadow Tables

For a table `vec_items` with vector column index `0`:

### 1. `__vec_items_ivf_centroids0`

Stores the k-means centroids. Empty until training.

```sql
CREATE TABLE IF NOT EXISTS "__vec_items_ivf_centroids0" (
  centroid_id INTEGER PRIMARY KEY,  -- 0-based centroid index
  centroid BLOB NOT NULL            -- full-precision centroid vector
);
```

- One row per centroid. `centroid_id` ranges from 0 to nlist-1.
- `centroid` is a raw float32 blob (dims * 4 bytes).
- When this table is empty, the index is in "flat mode."

### 2. `__vec_items_ivf_vectors0`

Stores vectors that have been assigned to a centroid.

```sql
CREATE TABLE IF NOT EXISTS "__vec_items_ivf_vectors0" (
  rowid INTEGER PRIMARY KEY,    -- matches the vec_items rowid
  centroid_id INTEGER NOT NULL, -- FK to centroids table
  vector BLOB NOT NULL          -- full-precision vector
);
```

- `centroid_id` is the index of the nearest centroid at assignment time.
- Indexed on `centroid_id` for efficient cell scans:

```sql
CREATE INDEX IF NOT EXISTS "__vec_items_ivf_vectors0_cell"
  ON "__vec_items_ivf_vectors0" (centroid_id);
```

### 3. `__vec_items_ivf_unassigned0`

Stores vectors inserted before centroids exist, or inserted after centroids
but not yet assigned (if we add a lazy-assign mode in the future).

```sql
CREATE TABLE IF NOT EXISTS "__vec_items_ivf_unassigned0" (
  rowid INTEGER PRIMARY KEY,  -- matches the vec_items rowid
  vector BLOB NOT NULL        -- full-precision vector
);
```

- On `compute-centroids` or `assign-vectors`, rows move from here to
  `_ivf_vectors`.
- At query time, if this table is non-empty, it is brute-force scanned and
  results are merged with the IVF probe results.

## Info Table Entries

Store IVF state in the existing `_info` shadow table:

| Key | Value | Description |
|-----|-------|-------------|
| `ivf_trained_0` | `0` or `1` | Whether centroids have been computed for column 0 |
| `ivf_nlist_0` | integer | Actual nlist used (important if nlist=0 at creation) |

## Shadow Table Name Macros

```c
#define VEC0_SHADOW_IVF_CENTROIDS_N_NAME  "__" TABLE "_ivf_centroids" NN
#define VEC0_SHADOW_IVF_VECTORS_N_NAME    "__" TABLE "_ivf_vectors" NN
#define VEC0_SHADOW_IVF_UNASSIGNED_N_NAME "__" TABLE "_ivf_unassigned" NN
```

(Follow the existing pattern from `VEC0_SHADOW_VECTORS_N_NAME`, etc.)

## Creation in `vec0Create`

In `vec0Create()`, after creating existing shadow tables, for each vector
column with `ivf.enabled`:

```c
for (int i = 0; i < p->numVectorColumns; i++) {
  if (!p->vector_columns[i].ivf.enabled) continue;

  // Create _ivf_centroids
  rc = sqlite3_exec(db, "CREATE TABLE ..._ivf_centroids...", ...);

  // Create _ivf_vectors with index
  rc = sqlite3_exec(db, "CREATE TABLE ..._ivf_vectors...", ...);
  rc = sqlite3_exec(db, "CREATE INDEX ..._ivf_vectors_cell...", ...);

  // Create _ivf_unassigned
  rc = sqlite3_exec(db, "CREATE TABLE ..._ivf_unassigned...", ...);

  // Set initial info entries
  rc = sqlite3_exec(db, "INSERT INTO _info VALUES('ivf_trained_0', '0')", ...);
}
```

## Prepared Statements

Prepare these statements in `vec0Create`/`vec0Connect` for each IVF column:

```c
// Centroid operations
"SELECT centroid FROM _ivf_centroids WHERE centroid_id = ?"
"INSERT OR REPLACE INTO _ivf_centroids(centroid_id, centroid) VALUES (?, ?)"
"SELECT centroid_id, centroid FROM _ivf_centroids"  // read all centroids
"DELETE FROM _ivf_centroids"  // clear before recompute

// Vector operations
"INSERT INTO _ivf_vectors(rowid, centroid_id, vector) VALUES (?, ?, ?)"
"SELECT rowid, vector FROM _ivf_vectors WHERE centroid_id = ?"  // cell scan
"DELETE FROM _ivf_vectors WHERE rowid = ?"

// Unassigned operations
"INSERT INTO _ivf_unassigned(rowid, vector) VALUES (?, ?)"
"SELECT rowid, vector FROM _ivf_unassigned"  // brute-force scan
"DELETE FROM _ivf_unassigned WHERE rowid = ?"
"SELECT count(*) FROM _ivf_unassigned"
```

## Cleanup in `vec0Destroy`

Drop all three shadow tables + the index when the virtual table is dropped.

## Files Changed

- `sqlite-vec-ivf.c`: Add `ivf_create_shadow_tables()`,
  `ivf_prepare_statements()`, `ivf_destroy_shadow_tables()`, name macros.
- `sqlite-vec.c`: ~3-line call to `ivf_create_shadow_tables()` in
  `vec0Create()`, ~2-line call to `ivf_destroy_shadow_tables()` in
  `vec0Destroy()`.

## Python Tests

```python
def test_ivf_shadow_tables_created(db):
    db.execute("""
        CREATE VIRTUAL TABLE t USING vec0(
            id integer primary key,
            v float[4] indexed by ivf(nlist=8)
        )
    """)
    # Verify shadow tables exist
    tables = [r[0] for r in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    assert "__t_ivf_centroids0" in tables
    assert "__t_ivf_vectors0" in tables
    assert "__t_ivf_unassigned0" in tables

    # Verify trained=0 initially
    trained = db.execute(
        "SELECT value FROM __t_info WHERE key='ivf_trained_0'"
    ).fetchone()[0]
    assert trained == '0'
```
