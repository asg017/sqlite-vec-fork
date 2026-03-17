# IVF Storage Redesign: Packed Blob Cells

## Problem

Current IVF stores one row per vector in `_ivf_vectors(rowid, centroid_id, vector)`.
Scanning a cell means reading N individual SQLite rows — each a separate B-tree
lookup. At 100k vectors with nlist=128, nprobe=16, that's ~12,500 row reads per
query.

vec0's chunk-based layout reads 1024 vectors in a single `sqlite3_blob_read()`.
This is why brute-force (71ms) beats IVF cell scan (136-191ms) at 100k.

## Solution

Store each IVF cell as a packed blob, exactly like vec0 chunks. One row per
centroid, with vectors packed contiguously in a BLOB column.

## New Shadow Table Schema

Replace the current three tables with two:

### `_ivf_centroids{NN}` (unchanged)

```sql
CREATE TABLE _ivf_centroids{NN} (
  centroid_id INTEGER PRIMARY KEY,
  centroid BLOB NOT NULL
);
```

### `_ivf_cells{NN}` (replaces `_ivf_vectors` + `_ivf_unassigned`)

```sql
CREATE TABLE _ivf_cells{NN} (
  cell_id    INTEGER PRIMARY KEY,  -- centroid_id, or -1 for unassigned
  n_vectors  INTEGER NOT NULL DEFAULT 0,
  validity   BLOB NOT NULL,        -- bitmap: which slots are live
  rowids     BLOB NOT NULL,        -- packed i64 array
  vectors    BLOB NOT NULL         -- packed float32 array
);
```

- **cell_id = centroid_id** for trained cells (0..nlist-1)
- **cell_id = -1** for the unassigned cell (flat mode vectors)
- Blob layout matches vec0 chunks exactly:
  - `validity`: `capacity / 8` bytes (1 bit per slot)
  - `rowids`: `capacity * 8` bytes (i64 per slot)
  - `vectors`: `capacity * dims * sizeof(float)` bytes

### Cell capacity management

Each cell starts with a fixed initial capacity (e.g., 256 slots). When full,
the cell is grown by doubling — read all blobs, realloc, write back. This is
rare after initial population.

Alternatively, capacity can be pre-computed during `compute-centroids` based
on the number of vectors assigned to each centroid, rounded up to the nearest
power of 2.

## Key Operations

### Insert (untrained / flat mode)

```
1. Read cell_id=-1 row (the unassigned cell)
2. Find next empty slot via validity bitmap (or append if full → grow)
3. sqlite3_blob_write() the vector at the slot offset
4. sqlite3_blob_write() the rowid at the slot offset
5. Set validity bit
6. Increment n_vectors
```

Same pattern as vec0's `vec0Update_InsertWriteFinalStep`, using
`sqlite3_blob_open()` for targeted writes at computed offsets.

### Insert (trained)

```
1. Scan all centroids (nlist distance computations) → find nearest
2. Read cell for that centroid_id
3. Write vector into next available slot (same as above)
```

### Delete

```
1. Find which cell contains the rowid:
   - If trained: scan cell blobs for matching rowid (or maintain a rowid→cell_id index)
   - Simpler: try each cell's rowids blob, or keep a separate lookup table
2. Clear validity bit at that slot
3. Decrement n_vectors
```

For delete, a small auxiliary index may help:

```sql
CREATE TABLE _ivf_rowid_map{NN} (
  rowid INTEGER PRIMARY KEY,
  cell_id INTEGER NOT NULL,
  slot INTEGER NOT NULL
);
```

This avoids scanning all cells on delete. 3 columns, tiny rows, fast lookup.

### KNN Query (cell scan)

The core performance win. For each probed cell:

```c
// One row read gets the entire cell
sqlite3_blob_open(db, schema, cells_table, "vectors", cell_id, 0, &blob);
sqlite3_blob_read(blob, vectorBuffer, n_vectors * vecSize, 0);

// Also read validity and rowids
// ... (same pattern)

// In-memory distance computation — identical to vec0Filter_knn_chunks_iter
for (int i = 0; i < capacity; i++) {
    if (!bitmap_get(validity, i)) continue;
    float dist = distance_fn(query, &vectors[i * D], D);
    // ... top-k insert
}
```

This is the exact same I/O pattern that makes vec0 brute-force fast:
**one blob read per cell, then pure in-memory iteration.**

### compute-centroids

```
1. Read all vectors from all cells (unassigned + any existing trained cells)
2. Run k-means → get nlist centroids
3. Assign each vector to nearest centroid → count per cell
4. Create new cell rows with pre-sized blobs:
   - capacity = round_up_pow2(count_per_cell)
   - validity = zeroblob(capacity/8), set bits for filled slots
   - rowids = packed i64 array
   - vectors = packed float array
5. Delete old cells, insert new ones
6. Delete unassigned cell (cell_id=-1) if all vectors are assigned
```

### assign-vectors (after manual centroid import)

Same as step 3-6 of compute-centroids, but without running k-means.

### clear-centroids

```
1. Read all vectors from all trained cells
2. Append them to the unassigned cell (cell_id=-1)
3. Delete trained cell rows
4. Delete centroids
```

## Eliminating Duplicate Storage

Currently IVF vectors are stored in BOTH the chunk table (because
`vec0Update_InsertWriteFinalStep` always runs) AND the IVF shadow tables.
This doubles DB size (689 MB vs 296 MB at 100k).

With the redesign, IVF-indexed vector columns should **skip chunk storage
entirely**. The IVF cells ARE the storage. Changes needed:

In `vec0Update_Insert()`:
```c
// Step #3: Write vectors to chunk — SKIP for IVF columns
for (int i = 0; i < p->numVectorColumns; i++) {
    if (p->vector_columns[i].ivf.enabled) continue;  // <-- NEW
    // ... existing chunk write logic
}
```

In `vec0Filter_knn()`:
```c
// Already dispatches to ivf_query_knn for IVF columns
// No chunk reads needed
```

In `vec0_get_vector_data()` (for point queries / xColumn):
```c
if (vector_column->ivf.enabled) {
    // Read from IVF cell via rowid_map lookup
    return ivf_get_vector_data(p, rowid, col_idx, outVector, outSize);
}
// ... existing chunk read path
```

This eliminates the 2x storage overhead.

## Skip _chunks and _rowids for IVF-only Tables

If ALL vector columns use IVF, the _chunks table and the chunk-position
columns in _rowids become unnecessary. We still need _rowids for rowid
management but can skip chunk_id/chunk_offset.

For v1, keep _chunks and _rowids as-is (they're small overhead). Optimize
later if needed.

## Expected Performance

At 100k, nlist=128, nprobe=16:

| Operation | Current (per-row) | Redesigned (packed blob) |
|-----------|-------------------|--------------------------|
| Cell scan I/O | ~780 row reads | 16 blob reads |
| Cell scan speed | ~136-191ms | ~10-15ms (estimated, proportional to vec0 chunk scan) |
| Insert (trained) | ~12k rows/s | ~50k rows/s (blob append vs row insert) |
| DB size | 689 MB (2x) | ~300 MB (no duplication) |

The 16 blob reads for nprobe=16 cells, each containing ~780 vectors
(100k/128), is comparable to vec0 scanning ~16 chunks of 1024 vectors
each. With the chunk-style distance computation in memory, query time
should approach vec0-flat proportional to nprobe/nlist.

## Implementation Steps

### Step A: Packed cell storage (core)

1. Replace `_ivf_vectors` + `_ivf_unassigned` with `_ivf_cells` table
2. Add `_ivf_rowid_map` for rowid→(cell_id, slot) lookup
3. Rewrite `ivf_insert()` to use blob write at computed offset
4. Rewrite `ivf_delete()` to clear validity bit via rowid_map lookup
5. Rewrite `ivf_query_knn()` to use single blob read per cell + in-memory scan

### Step B: Eliminate duplicate storage

6. Skip chunk writes in `vec0Update_InsertWriteFinalStep` for IVF columns
7. Add `ivf_get_vector_data()` for point queries (xColumn)
8. Update `vec0Update_Delete` to skip chunk validity clearing for IVF columns

### Step C: Centroid commands on new storage

9. Rewrite `compute-centroids` to read from / write to packed cells
10. Rewrite `assign-vectors` and `clear-centroids`
11. Update `ivf_create_shadow_tables` / `ivf_drop_shadow_tables`

### Step D: Cell capacity management

12. Implement cell growth (double capacity when full)
13. Pre-size cells during `compute-centroids` based on assignment counts
14. Optional: compact cells on `optimize` command (reclaim deleted slots)

## Compatibility

This is a breaking change to the IVF shadow table schema. Since IVF is new
and unreleased, there's no migration concern. The old `_ivf_vectors` /
`_ivf_unassigned` tables are simply replaced.
