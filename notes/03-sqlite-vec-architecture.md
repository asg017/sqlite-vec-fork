# sqlite-vec Architecture for Index Integration

Source: `/home/sprite/sqlite-vec-private/sqlite-vec.c` (~12,000 lines)

## vec0 Virtual Table Structure

### Column Types
- **Vector columns** (up to 16): `embedding float[768] distance_metric=cosine`
- **Partition columns** (up to 4): `user_id integer partition key`
- **Auxiliary columns** (up to 16): `+description text`
- **Metadata columns** (up to 16): `category text` (filterable in KNN)
- **Hidden columns**: `distance` (output), `k` (parameter)

### INDEXED BY Syntax

Vector columns can specify an index type:

```sql
embedding float[768] distance_metric=cosine INDEXED BY diskann(n_neighbors=72, ...)
embedding float[768] distance_metric=cosine INDEXED BY ivf(nlist=128, nprobe=16, ...)
```

Parsing flow: `vec0_parse_vector_column()` tokenizes the column definition,
recognizes `INDEXED BY`, then dispatches to index-specific parsers.

## Shadow Tables

### Core (always present)
| Table | Purpose |
|-------|---------|
| `{t}_rowids` | rowid ↔ chunk mapping (chunk_id, chunk_offset) |
| `{t}_chunks` | Packed vectors in 64-item chunks (validity bitmap, rowids, data) |
| `{t}_vector_chunks{NN}` | Per-column chunk vector blobs |
| `{t}_auxiliary` | Auxiliary column values |
| `{t}_metadatachunks{NN}` | Metadata column values per chunk |

### DiskANN shadow tables
| Table | Purpose |
|-------|---------|
| `{t}_vectors{NN}` | Full-precision vectors (1 row per vector) |
| `{t}_diskann_nodes{NN}` | Graph nodes: neighbor IDs + quantized neighbor vectors |
| `{t}_diskann_buffer{NN}` | Staging area for batched inserts |
| `{t}_diskann_medoid{NN}` | Single row: entry point rowid |

### IVF shadow tables
| Table | Purpose |
|-------|---------|
| `{t}_ivf_centroids{NN}` | K-means centroids |
| `{t}_ivf_cells{NN}` | Packed vectors per centroid (64 max per row) |
| `{t}_ivf_rowid_map{NN}` | rowid → cell location for O(1) delete |
| `{t}_ivf_vectors{NN}` | Full-precision vectors for re-ranking |

## Index Integration Pattern

### Step 1: Configuration Struct
```c
struct Vec0DiskannConfig {
    int enabled;
    enum quantizer_type;
    int n_neighbors;
    int search_list_size;
    float alpha;
    int buffer_threshold;
};
```
Stored in `vec0_vtab` per vector column.

### Step 2: Parsing
Extend `vec0_parse_vector_column()` to recognize `INDEXED BY annoy(...)`.
Parse parameters into config struct.

### Step 3: Shadow Table Creation
In `vec0_init()` (xCreate path), CREATE the shadow tables.
Register names in `vec0ShadowName()`.

### Step 4: Prepared Statements
Cache prepared statements on `vec0_vtab` for all shadow table operations
(read, write, delete). Avoids per-operation prepare/finalize overhead.

### Step 5: Insert Hook
In `vec0Update_Insert()` (line ~10737), after writing chunk data:
```c
if (column has annoy index) {
    annoy_insert(vtab, col_idx, rowid, vector);
}
```

### Step 6: Delete Hook
In `vec0Update_Delete()`:
```c
if (column has annoy index) {
    annoy_delete(vtab, col_idx, rowid);
}
```

### Step 7: Query Hook
In `vec0Filter()` for KNN plan, dispatch to annoy search:
```c
if (column has annoy index) {
    vec0Filter_knn_annoy(cursor, col_idx, query_vector, k, ...);
}
```

### Step 8: BestIndex
Annoy-indexed columns should report lower cost for KNN plans.

## Query Execution Flow

```
vec0BestIndex() → determine plan (KNN/POINT/FULLSCAN), encode in idxStr
vec0Filter()    → parse idxStr, dispatch to appropriate search
vec0Next()      → iterate results
vec0Column()    → return distance, vectors, metadata
vec0Eof()       → signal done
```

### idxStr Encoding
1-byte header (plan type) + 4-byte blocks per constraint:
- `{` = KNN match vector
- `}` = k limit
- `[` = rowid IN filter
- `]` = partition constraint
- `&` = metadata constraint

## File Organization Pattern

Both DiskANN and IVF use separate `.c` files `#include`d into `sqlite-vec.c`:

```
sqlite-vec.c               Main file, ~50 lines of additions for each index
sqlite-vec-diskann.c       DiskANN-specific logic
sqlite-vec-ivf.c           IVF-specific logic (with separate kmeans file)
```

This keeps the main file manageable and index logic self-contained.
