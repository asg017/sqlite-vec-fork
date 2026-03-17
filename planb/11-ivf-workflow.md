# Complete End-to-End Description of the IVF Workflow in sqlite-vec

## Overview

The IVF (Inverted File Index) implementation in sqlite-vec is a fixed-size cell-based index for approximate nearest neighbor search. It organizes vectors into clusters (centroids) and stores them in fixed-size cells capped at 64 vectors each to avoid expensive multi-page blob traversal.

## 1. Table Creation

When executing `CREATE VIRTUAL TABLE t USING vec0(v float[768] indexed by ivf(nlist=128, nprobe=16))`:

### Parser Flow
- `vec0_parse_ivf_options()` parses the IVF parameters: `nlist` (number of centroids, default 128) and `nprobe` (number of centroids to probe, default 10)
- If nprobe is not explicitly specified, it defaults to nprobe
- Validation: nprobe is clamped to nlist if nprobe > nlist

### Schema Configuration
The IVF config is stored in `VectorColumnDefinition.ivf` with fields:
- `enabled`: Whether IVF is active
- `nlist`: Number of centroids
- `nprobe`: Number of centroids to probe during queries

### Shadow Table Creation
When a table is created with IVF enabled, `ivf_create_shadow_tables()` creates three tables per IVF vector column:

#### 1. `_ivf_centroids%02d` (where %02d = column index)
```sql
CREATE TABLE _ivf_centroids00 (
  centroid_id INTEGER PRIMARY KEY,
  centroid BLOB NOT NULL
)
```
- Stores the actual centroid vectors (nlist rows)
- Each row is a D-dimensional float vector packed as BLOB

#### 2. `_ivf_cells%02d`
```sql
CREATE TABLE _ivf_cells00 (
  centroid_id INTEGER NOT NULL,
  n_vectors INTEGER NOT NULL DEFAULT 0,
  validity BLOB NOT NULL,           -- Bitfield: cap/8 bytes
  rowids BLOB NOT NULL,              -- Fixed array: cap * 8 bytes
  vectors BLOB NOT NULL              -- Packed vectors: cap * D * 4 bytes
)
CREATE INDEX _ivf_cells00_centroid ON _ivf_cells00 (centroid_id)
```
- Multiple cell rows per centroid (cells overflow when full)
- `rowid`: Auto-increment cell ID
- `centroid_id`: Foreign key to centroids table
- `n_vectors`: Current number of vectors in cell (0 to VEC0_IVF_CELL_MAX_VECTORS=64)
- `validity`: Bitfield (1 bit per slot) marking valid/deleted vectors
- `rowids`: Fixed-size array storing the table rowids of inserted vectors
- `vectors`: Packed float array of all vectors in the cell

#### 3. `_ivf_rowid_map%02d`
```sql
CREATE TABLE _ivf_rowid_map00 (
  rowid INTEGER PRIMARY KEY,
  cell_id INTEGER NOT NULL,
  slot INTEGER NOT NULL
)
```
- Maps each vector's rowid to its cell location and slot
- Used for fast deletion: given a rowid, find which cell and slot to invalidate

#### 4. `_info` table (shared, not IVF-specific)
```sql
INSERT INTO _info (key, value) VALUES ('ivf_trained_0', '0')
```
- Tracks training state per vector column
- `ivf_trained_0` = 0 (untrained), 1 (trained)
- Cached in memory: `p->ivfTrainedCache[col_idx]`

### Constants
- `VEC0_IVF_CELL_MAX_VECTORS = 64`: Max vectors per cell (~200KB per cell at 768 dims)
- `VEC0_IVF_DEFAULT_NLIST = 128`: Default number of centroids
- `VEC0_IVF_DEFAULT_NPROBE = 10`: Default number of centroids to probe
- `VEC0_IVF_UNASSIGNED_CENTROID_ID = -1`: Special ID for untrained inserts

## 2. Insert (Untrained/Flat Mode)

When inserting `INSERT INTO t(id, v) VALUES (1, ?)` before centroids are computed:

### Call Path
```
vec0Update()
  → ivf_handle_command() [checks for special commands, returns SQLITE_EMPTY]
  → vec0Update_Insert()
    → vec0Update_InsertRowidStep() [allocate rowid]
    → vec0Update_InsertNextAvailableStep() [find chunk position]
    → vec0Update_InsertWriteFinalStep() [write to chunk tables]
    → ivf_insert()
      → ivf_is_trained() [checks ivf_trained_0 in _info table]
      → ivf_cell_insert()
        → ivf_cell_find_or_create()
```

### ivf_insert() in Untrained Mode
```c
static int ivf_insert(vec0_vtab *p, int col_idx, i64 rowid,
                      const void *vectorData, int vectorSize) {
  if (!ivf_is_trained(p, col_idx)) {
    return ivf_cell_insert(p, col_idx, VEC0_IVF_UNASSIGNED_CENTROID_ID,
                            rowid, vectorData, vectorSize);
  }
```

- Checks `p->ivfTrainedCache[col_idx]` (memoized in-memory cache)
- If untrained, inserts directly to cells with `centroid_id = -1`
- Trained mode skipped

### ivf_cell_insert() Details

1. **Find or Create Cell**: `ivf_cell_find_or_create()`
   - Queries: `SELECT rowid, n_vectors FROM _ivf_cells WHERE centroid_id = ? AND n_vectors < 64 LIMIT 1`
   - Uses cached statement `p->stmtIvfCellMeta[col_idx]`
   - If no cell has space, creates a new one via `ivf_cell_create()`

2. **Create New Cell**: `ivf_cell_create()`
   ```sql
   INSERT INTO _ivf_cells (centroid_id, n_vectors, validity, rowids, vectors)
     VALUES (?, 0, zeroblob(8), zeroblob(512), zeroblob(196608))
   ```
   - `zeroblob(64/8)` for validity (8 bytes for 64 slots)
   - `zeroblob(64 * 8)` for rowids (512 bytes for 64 i64s)
   - `zeroblob(64 * 768 * 4)` for vectors (196,608 bytes at 768-dim float32)
   - Returns auto-increment cell_id

3. **Write to Cell Blobs** (3 separate blob writes per insert):
   ```c
   // Set validity bit (slot position)
   sqlite3_blob_open(db, schema, "cells", "validity", cell_id, 1, &blob)
   sqlite3_blob_read(&blob, &bx, 1, slot / 8)
   bx |= (1 << (slot % 8))
   sqlite3_blob_write(&blob, &bx, 1, slot / 8)
   sqlite3_blob_close(&blob)

   // Write rowid
   sqlite3_blob_open(db, schema, "cells", "rowids", cell_id, 1, &blob)
   sqlite3_blob_write(&blob, &rowid, 8, slot * 8)
   sqlite3_blob_close(&blob)

   // Write vector
   sqlite3_blob_open(db, schema, "cells", "vectors", cell_id, 1, &blob)
   sqlite3_blob_write(&blob, vectorData, vecSize, slot * vecSize)
   sqlite3_blob_close(&blob)
   ```
   - 3 `sqlite3_blob_open()`/`write()`/`close()` calls per insert (**hot path bottleneck**)

4. **Update Cell Metadata**:
   ```sql
   UPDATE _ivf_cells SET n_vectors = n_vectors + 1 WHERE rowid = ?
   ```

5. **Insert Rowid Map**:
   ```sql
   INSERT INTO _ivf_rowid_map (rowid, cell_id, slot) VALUES (?, ?, ?)
   ```

### Chunk Table Writes (Parallel Path)
The standard vec0 pipeline also writes to `_chunks` shadow table:
- `vec0Update_InsertNextAvailableStep()`: Finds chunk position
- `vec0Update_InsertWriteFinalStep()`: 
  - Skips writing vector columns with `ivf.enabled` (line 8167)
  - Writes to rowids column of `_chunks` table
  - Writes metadata
  - **Note**: This is the 52% bottleneck in insert profiling

## 3. Insert (Trained Mode)

After `INSERT INTO t(id) VALUES ('compute-centroids')`, inserts work differently:

### ivf_insert() in Trained Mode
```c
if (ivf_is_trained(p, col_idx)) {
  // Find nearest centroid (O(nlist * dims) distance computations)
  int best_centroid = -1;
  float min_dist = FLT_MAX;
  
  sqlite3_stmt *stmt = p->stmtIvfCentroidsAll[col_idx];
  sqlite3_reset(stmt);
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    int cid = sqlite3_column_int(stmt, 0);
    const float *c = (const float *)sqlite3_column_blob(stmt, 1);
    float dist = ivf_l2_dist((const float *)vectorData, c, D);
    if (dist < min_dist) { min_dist = dist; best_centroid = cid; }
  }
  
  return ivf_cell_insert(p, col_idx, best_centroid, rowid, vectorData, vectorSize);
}
```

Key differences from untrained mode:
1. **Centroid Scan**: Every insert scans all `nlist` centroids
2. **Distance Computation**: `ivf_l2_dist()` calculates squared L2 distance between query vector and each centroid
3. **Cell Assignment**: Vector is assigned to the nearest centroid's cells
4. **Bottleneck**: Centroid scan is O(nlist × dims) per insert (11% of insert time at 100k vectors)

The `ivf_l2_dist()` function from kmeans.c:
```c
static float ivf_l2_dist(const float *a, const float *b, int D) {
  float sum = 0.0f;
  for (int d = 0; d < D; d++) {
    float diff = a[d] - b[d];
    sum += diff * diff;
  }
  return sum;
}
```

The rest of `ivf_cell_insert()` is identical to untrained mode.

## 4. compute-centroids Command

`INSERT INTO t(id) VALUES ('compute-centroids')` triggers special handling:

### Command Dispatch
```c
vec0Update() → ivf_handle_command(p, "compute-centroids", ...)
  → ivf_cmd_compute_centroids(p, col_idx, 0, max_iter=25, seed=0)
```

### Step 1: Load All Vectors
```c
ivf_load_all_vectors(p, col_idx, &vectors, &rowids, &N)
```

- Iterates through all `_ivf_cells` rows
- For each cell, reads `n_vectors`, `validity`, `rowids`, `vectors` blobs
- Extracts only valid vectors (bits set in validity bitfield)
- Returns flattened arrays: `vectors[N*D]`, `rowids[N]`, count `N`

Example cell unpacking:
```c
while (sqlite3_step(stmt) == SQLITE_ROW) {
  int n = sqlite3_column_int(stmt, 0);
  const unsigned char *val = sqlite3_column_blob(stmt, 1);  // validity
  const i64 *rids = sqlite3_column_blob(stmt, 2);           // rowids
  const float *vecs = sqlite3_column_blob(stmt, 3);        // vectors
  for (int i = 0; i < cap; i++) {
    if (val[i / 8] & (1 << (i % 8))) {  // if valid
      rowids[idx] = rids[i];
      memcpy(&vectors[idx * D], &vecs[i * D], vecSize);
      idx++;
    }
  }
}
```

### Step 2: K-Means Clustering
```c
ivf_kmeans(vectors, N, D, nlist, max_iter=25, seed=0, out_centroids)
```

From `sqlite-vec-ivf-kmeans.c`, implements Lloyd's algorithm:

1. **K-means++ Initialization**: `ivf_kmeans_init_plusplus()`
   - Picks first centroid randomly
   - Each subsequent centroid chosen with probability ∝ (distance to nearest centroid)²
   - Uses xorshift32 PRNG

2. **Lloyd's Iteration** (up to 25 iterations):
   - **Assignment**: Find nearest centroid for each vector
   - **Update**: Recompute centroids as mean of assigned vectors
   - **Convergence**: Stop when no vectors change assignment
   - **Empty cluster handling**: Reassign empty clusters to farthest point

### Step 3: Compute Assignments
```c
for (int i = 0; i < N; i++)
  assignments[i] = ivf_nearest_centroid(&vectors[i * D], centroids, D, nlist);
```

### Step 4: Clear and Rebuild Index
```c
sqlite3_exec(db, "SAVEPOINT ivf_train");

DELETE FROM _ivf_centroids
DELETE FROM _ivf_cells
DELETE FROM _ivf_rowid_map
```

### Step 5: Write Centroids
```sql
INSERT INTO _ivf_centroids (centroid_id, centroid) VALUES (0, blob), (1, blob), ...
```

### Step 6: Build Cells (Fixed-Size Chunking)
For each centroid c:
1. Collect all vectors with `assignments[i] == c`
2. Partition into fixed-size chunks of 64 vectors max
3. For each chunk, create a cell row:
   ```sql
   INSERT INTO _ivf_cells (centroid_id, n_vectors, validity, rowids, vectors)
     VALUES (c, n_vectors, validity_blob, rowids_blob, vectors_blob)
   ```
4. For each vector in the cell, insert rowid map:
   ```sql
   INSERT INTO _ivf_rowid_map (rowid, cell_id, slot) VALUES (rowid, cell_id, slot)
   ```

The cell building uses a tight loop to minimize allocations:
```c
for (int c = 0; c < nlist; c++) {
  int slot = 0;
  for (int i = 0; i < N; i++) {
    if (assignments[i] != c) continue;
    if (slot >= 64) {
      // Flush current cell, create new one
      sqlite3_step(stmtCell);
      i64 flushed_cell_id = sqlite3_last_insert_rowid(db);
      for (int s = 0; s < slot; s++) {
        // Insert rowid_map entries
      }
      slot = 0;
    }
    val[slot / 8] |= (1 << (slot % 8));
    rids[slot] = rowids[i];
    memcpy(&vecs[slot * D], &vectors[i * D], vecSize);
    slot++;
  }
  // Flush remaining vectors in centroid
}
```

### Step 7: Mark Trained
```sql
INSERT OR REPLACE INTO _info (key, value) VALUES ('ivf_trained_0', '1')
```
- Sets `p->ivfTrainedCache[col_idx] = 1`

### Step 8: Cleanup
```c
sqlite3_exec(db, "RELEASE ivf_train");
sqlite3_free(vectors); sqlite3_free(rowids); sqlite3_free(centroids); sqlite3_free(assignments);
```

## 5. KNN Query

Executing `SELECT id FROM t WHERE v MATCH ? AND k=10`:

### Call Path
```
vec0Filter()
  → vec0Filter_knn() [idxNum contains vector column index]
    → [Extract query vector and k from argv]
    → [if vector_column->ivf.enabled]
      → ivf_query_knn(p, col_idx, queryVector, k, knn_data)
        → [Find top nprobe centroids by distance]
        → ivf_scan_cells_from_stmt()
        → qsort(candidates)
        → Return top k candidates
```

### ivf_query_knn() in Trained Mode

1. **Load Centroids**:
   ```sql
   SELECT centroid_id, centroid FROM _ivf_centroids
   ```
   - Cached statement: `p->stmtIvfCentroidsAll[col_idx]`
   - Loaded into memory array: `cd[nlist]` with `{id, distance}`

2. **Find Top nprobe Centroids**:
   ```c
   float min_dist = FLT_MAX;
   for each centroid:
     dist = ivf_l2_dist(queryVector, centroid, D)
   // Selection sort to find top nprobe
   for (int i = 0; i < nprobe; i++) {
     // Bubble smallest distances to front
   }
   ```

3. **Build SQL Query**:
   ```c
   "SELECT n_vectors, validity, rowids, vectors FROM _ivf_cells00 
    WHERE centroid_id IN (5, 12, 3, ...) OR centroid_id = -1"
   ```
   - Includes probed centroids PLUS unassigned cells (centroid_id = -1)
   - Scans all these cells in one query

4. **Scan Cells**: `ivf_scan_cells_from_stmt()`
   ```c
   while (sqlite3_step(stmt) == SQLITE_ROW) {
     int n = sqlite3_column_int(stmt, 0);
     const unsigned char *val = sqlite3_column_blob(stmt, 1);  // validity
     const i64 *rids = sqlite3_column_blob(stmt, 2);           // rowids
     const float *vecs = sqlite3_column_blob(stmt, 3);        // vectors
     
     for (int i = 0; i < cap; i++) {
       if (val[i / 8] & (1 << (i % 8))) {  // if valid
         float dist = ivf_l2_dist(queryVector, &vecs[i * D], D);
         // Append to candidates array if promising
         candidates[nCandidates++] = {.rowid = rids[i], .distance = dist};
       }
     }
   }
   ```

5. **Sort Candidates**:
   ```c
   qsort(candidates, nCandidates, sizeof(IvfCandidate), ivf_candidate_cmp);
   ```
   - Sorts by distance ascending

6. **Return Top k**:
   ```c
   knn_data->k_used = min(k, nCandidates);
   for (i = 0; i < k_used; i++) {
     knn_data->rowids[i] = candidates[i].rowid;
     knn_data->distances[i] = candidates[i].distance;
   }
   ```

### ivf_query_knn() in Untrained Mode

```c
if (!trained) {
  // Flat mode: scan only unassigned cells
  SELECT ... FROM _ivf_cells00 WHERE centroid_id = -1
  // Rest is same: load vectors, compute distances, sort, return top k
}
```

### Performance Characteristics
- **Distance computation**: 49% of query time (scalar L2 loop)
- **Disk I/O**: 41% of query time (reading cell overflow pages via `accessPayload`)
- **Sorting**: 9% of query time (qsort over all candidates)
- **Centroid scan**: <1% (cached statements, small nlist)

## 6. Shadow Tables Summary

Per IVF vector column (indexed by col_idx):

| Table Name | Schema | Purpose |
|------------|--------|---------|
| `_ivf_centroids%02d` | `centroid_id PK, centroid BLOB` | Stores k centroid vectors (nlist rows) |
| `_ivf_cells%02d` | `centroid_id, n_vectors, validity BLOB, rowids BLOB, vectors BLOB` | Fixed-size cells (64 max vectors each) organized by centroid. Multiple rows per centroid. Index on centroid_id. |
| `_ivf_rowid_map%02d` | `rowid PK, cell_id, slot` | Maps each vector's original rowid to its cell location for fast deletion |
| `_info` | `key, value` | Shared table: `ivf_trained_<col_idx>` flag (0=untrained, 1=trained) |

Plus standard vec0 tables (not IVF-specific):
| Table Name | Schema | Purpose |
|------------|--------|---------|
| `_chunks` | Contains packed vector columns (skipped if all columns have IVF) | |
| `_rowids` | Maps user IDs/primary keys to internal rowids | |
| `_auxiliary` | Stores auxiliary columns | |
| `_metadatachunks%02d` | Stores metadata columns | |

## 7. Bottlenecks and Optimization Opportunities

Based on `/Users/alex/projects/sqlite-vec/ivf-benchmarks/TODO.md`:

### Insert Profiling (10k inserts into 30k trained IVF, 100k vectors total)

| Component | % of Time | Cause |
|-----------|-----------|-------|
| vec0 chunk writes | 52% | `vec0Update_InsertWriteFinalStep` writes to vector chunk tables (which IVF doesn't use for queries). These writes are wasteful for IVF-only columns. |
| IVF cell blob writes | 28% | `ivf_cell_insert` makes 3 separate `sqlite3_blob_open/write/close` calls per insert (validity, rowids, vectors) |
| Centroid scan | 11% | `ivf_insert` scans all centroids O(nlist × dims) to find nearest |
| Rowid/overhead | 9% | `_rowids` table updates, `rowid_map` inserts |

### Priority Optimizations

**HIGH PRIORITY** (2× insert speedup):
1. **Skip chunk writes for IVF columns**: In `vec0Update_InsertWriteFinalStep`, skip vector columns with `ivf.enabled`. Also skip `vec0Update_InsertNextAvailableStep` for IVF-only tables.

**MEDIUM PRIORITY** (11% speedup):
2. **Cache centroids in memory**: Load centroids once into float array on first trained insert, invalidate on `compute-centroids`. Avoids re-reading centroid blobs.

**MEDIUM PRIORITY** (3% speedup):
3. **Batch blob writes**: Keep blob handles open across inserts to same cell, close on cell fill or commit.

### Query Profiling (200 queries on 100k trained IVF, nlist=128, nprobe=16)

| Component | % of Time | Cause |
|-----------|-----------|-------|
| Distance computation | 49% | `ivf_l2_dist` is scalar C loop in `ivf_scan_cells_from_stmt` |
| Disk I/O | 41% | `pread` via `accessPayload` reading cell overflow pages |
| Candidate sorting | 9% | `qsort` over all candidates |
| Centroid scan | <1% | Finding top nprobe centroids |

### Query Optimization Priority

**HIGH PRIORITY**:
1. **SIMD distance function**: Replace scalar `ivf_l2_dist` with NEON intrinsics (already available in sqlite-vec: `distance_l2_sqr_float()`)

**HIGH PRIORITY**:
2. **Direct blob reads**: Use `sqlite3_blob_read` instead of VDBE query for cell scans to bypass overflow page traversal overhead

**MEDIUM PRIORITY**:
3. **Top-k heap**: Replace `qsort` with min-heap of size k to avoid O(n log n) sort

**MEDIUM PRIORITY**:
4. **Parallel cell reads**: Pre-sort cells by rowid to improve I/O locality

### Training Optimizations

- **Train on larger samples**: Current recall ~0.89 from training on 8× nlist vectors. Need 30-64× nlist for better clusters.
- **SIMD k-means**: Inner loop of `ivf_nearest_centroid` and k-means has same scalar distance computation
- **Mini-batch k-means**: Converge faster with similar quality

### Storage Optimization

- **Eliminate double storage**: IVF vectors stored in both chunk tables (296 MB) and IVF cells (296 MB). Skipping chunks would halve storage at 100k vectors.

### Benchmark Context

At 1M vectors:
- `vec0-bit` (brute force bit-quantized): 19s insert, 101ms query, 0.918 recall
- `vec0-int8`: 18s insert, 217ms query, 0.998 recall
- `ivf-n256-p8`: 449s insert, 173ms query, 0.872 recall (best IVF query)
- `vec0-flat` (full brute force): 18s insert, 706ms query, 1.000 recall

IVF wins at smaller datasets (100k: 10ms vs 19ms for bit) but slower at 1M due to unoptimized implementation.

---

## Key Data Structures

### vec0_ivf_config (in VectorColumnDefinition)
```c
struct Vec0IvfConfig {
  int enabled;    // Whether IVF is active
  int nlist;      // Number of centroids (clusters)
  int nprobe;     // Number of centroids to probe during queries
};
```

### IvfCandidate (used during query)
```c
struct IvfCandidate {
  i64 rowid;
  float distance;
};
```

### IvfCentroidDist (used during centroid selection)
```c
struct IvfCentroidDist {
  int id;
  float dist;
};
```

### Cached Statements in vec0_vtab
```c
sqlite3_stmt *stmtIvfCellMeta[VEC0_MAX_VECTOR_COLUMNS];        // Find cell with space
sqlite3_stmt *stmtIvfCentroidsAll[VEC0_MAX_VECTOR_COLUMNS];    // Load all centroids
sqlite3_stmt *stmtIvfCellUpdateN[VEC0_MAX_VECTOR_COLUMNS];     // Increment n_vectors
sqlite3_stmt *stmtIvfRowidMapInsert[VEC0_MAX_VECTOR_COLUMNS];  // Insert rowid_map
sqlite3_stmt *stmtIvfRowidMapLookup[VEC0_MAX_VECTOR_COLUMNS];  // Find cell from rowid
sqlite3_stmt *stmtIvfRowidMapDelete[VEC0_MAX_VECTOR_COLUMNS];  // Delete from rowid_map
int ivfTrainedCache[VEC0_MAX_VECTOR_COLUMNS];                  // Cached trained flag (-1=unknown)
char *shadowIvfCellsNames[VEC0_MAX_VECTOR_COLUMNS];            // Cell table name
```

---

## File Locations

- `/Users/alex/projects/sqlite-vec/sqlite-vec-ivf.c`: Full IVF implementation (1080 lines)
- `/Users/alex/projects/sqlite-vec/sqlite-vec-ivf-kmeans.c`: K-means algorithm (215 lines)
- `/Users/alex/projects/sqlite-vec/sqlite-vec.c`: Main vtab implementation (10081 lines)
  - Lines 4756-4758: IVF includes
  - Lines 4761+: `vec0_init()` - table creation
  - Lines 7097+: `vec0Filter_knn()` - query dispatch
  - Lines 7329-7340: IVF query dispatch
  - Lines 8426+: `vec0Update_Insert()` - insert flow
  - Lines 8571-8579: IVF insert dispatch
- `/Users/alex/projects/sqlite-vec/ivf-benchmarks/TODO.md`: Profiling results and optimization roadmap