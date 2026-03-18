# Rescore TODO

## Metadata filtering (broken — silent no-op)

Rescore indexes currently ignore metadata filtering constraints during KNN queries (`(void)aMetadataIn;` at rescore.c:362). Queries like `WHERE embedding MATCH ?1 AND genre = 'rock'` silently skip the genre filter.

### What works already

- **Partition keys** — `vec0_chunks_iter()` applies partition WHERE clauses before rescore sees chunks (shared code path)
- **Auxiliary columns** — retrieved via `vec0_get_auxiliary_value_for_rowid()` from `_auxiliary` shadow table, keyed by rowid (independent of query path)

### Fix

Port metadata filtering logic from `vec0Filter_knn_chunks_iter()` (sqlite-vec.c:7249-7277) into `rescore_knn()` (sqlite-vec-rescore.c). The normal path opens metadata blob per chunk, calls `vec0_set_metadata_filter_bitmap()`, and ANDs the result into the candidate bitmap. Rescore needs the same in its chunk loop.

#### 1. Remove `(void)aMetadataIn;` (line 362)

#### 2. Add `hasMetadataFilters` detection before the chunk loop (~line 424)

Same pattern as normal path (sqlite-vec.c:7148-7158): scan idxStr for `VEC0_IDXSTR_KIND_METADATA_CONSTRAINT` entries.

```c
int hasMetadataFilters = 0;
for (int i = 0; i < argc; i++) {
  int idx = 1 + (i * 4);
  if (idxStr[idx] == VEC0_IDXSTR_KIND_METADATA_CONSTRAINT)
    hasMetadataFilters = 1;
}
```

#### 3. Allocate metadata bitmap and blob array (if needed)

```c
u8 *bmMetadata = NULL;
sqlite3_blob *metadataBlobs[VEC0_MAX_METADATA_COLUMNS] = {0};
if (hasMetadataFilters) {
  bmMetadata = sqlite3_malloc(p->chunk_size / CHAR_BIT);
  if (!bmMetadata) { rc = SQLITE_NOMEM; goto cleanup; }
}
```

#### 4. Inside chunk loop, after rowid-in filtering (after line 455), add metadata filtering

Mirrors sqlite-vec.c:7249-7277 exactly:

```c
if (hasMetadataFilters) {
  for (int i = 0; i < argc; i++) {
    int idx = 1 + (i * 4);
    if (idxStr[idx] != VEC0_IDXSTR_KIND_METADATA_CONSTRAINT)
      continue;
    int metadata_idx = idxStr[idx + 1] - 'A';
    int operator = idxStr[idx + 2];

    if (!metadataBlobs[metadata_idx]) {
      rc = sqlite3_blob_open(p->db, p->schemaName,
               p->shadowMetadataChunksNames[metadata_idx],
               "data", chunk_id, 0, &metadataBlobs[metadata_idx]);
      if (rc != SQLITE_OK) goto cleanup;
    } else {
      rc = sqlite3_blob_reopen(metadataBlobs[metadata_idx], chunk_id);
      if (rc != SQLITE_OK) goto cleanup;
    }

    bitmap_clear(bmMetadata, p->chunk_size);
    rc = vec0_set_metadata_filter_bitmap(p, metadata_idx, operator,
             argv[i], metadataBlobs[metadata_idx], chunk_id,
             bmMetadata, p->chunk_size, aMetadataIn, i);
    if (rc != SQLITE_OK) goto cleanup;
    bitmap_and_inplace(b, bmMetadata, p->chunk_size);
  }
}
```

Note: use `sqlite3_blob_reopen` for subsequent chunks (same metadata column) instead of close+reopen — more efficient than the normal path.

#### 5. Cleanup: close metadata blobs and free bitmap

In the `cleanup:` section:

```c
for (int i = 0; i < VEC0_MAX_METADATA_COLUMNS; i++) {
  if (metadataBlobs[i]) sqlite3_blob_close(metadataBlobs[i]);
}
sqlite3_free(bmMetadata);
```

### Tests to add (tests/test-rescore.py)

- `test_knn_with_metadata_filter` — create rescore table with a metadata column (e.g., `category text`), insert vectors with different categories, run KNN with `WHERE category = 'a'`, verify only matching rows returned
- `test_knn_with_metadata_int_filter` — same with integer metadata and range operators (>, <, =)
- `test_knn_with_metadata_and_partition` — combine partition key + metadata filter + rescore KNN
- `test_knn_metadata_in_operator` — test `WHERE category IN ('a', 'b')` with rescore

### Reference: how the normal path works

The dispatch point is in `vec0Filter_knn()` (sqlite-vec.c:7676-7687):

```c
if (vector_column->rescore.enabled) {
  rc = rescore_knn(p, pCur, vector_column, vectorColumnIdx, arrayRowidsIn,
                   aMetadataIn, idxStr, argc, argv, queryVector, k, knn_data);
  // ...
}
```

The normal (non-rescore) path in `vec0Filter_knn_chunks_iter()` (sqlite-vec.c:7249-7277):
- For each chunk, iterates metadata constraints from idxStr
- Opens metadata chunk blob: `sqlite3_blob_open(..., shadowMetadataChunksNames[metadata_idx], "data", chunk_id, ...)`
- Calls `vec0_set_metadata_filter_bitmap()` which reads the blob, applies the operator, sets bits in bitmap
- ANDs bitmap into candidate bitmap `b` via `bitmap_and_inplace(b, bmMetadata, chunk_size)`

idxStr encoding for metadata: `& + [metadata_idx as 'A'+idx] + [operator] + '_'`
- Operators: EQ='a', GT='b', LE='c', LT='d', GE='e', NE='f', IN='g'

`vec0_set_metadata_filter_bitmap` is non-static (declared at sqlite-vec.c:6871) and accessible from rescore.c since it's `#include`d at sqlite-vec.c:7442.

### Reference: metadata storage

Metadata is stored in chunk-aligned blobs in `_metadatachunks{XX}` shadow tables:
- BOOLEAN: 1 bit per value (packed)
- INTEGER: 8 bytes (i64) per value
- FLOAT: 8 bytes (double) per value
- TEXT: 16-byte view (4-byte length + 12 bytes inline) + overflow in `_metadatatext{XX}` table

## NEON min_idx optimization (backburner)

`min_idx` is O(n*k) with repeated linear scans — 30.5% of rescore query time at 10k vectors. Current implementation (sqlite-vec.c:6249-6279) does k passes over n elements. For k_oversample=80, chunk_size=256, that's 80 x 256 = 20,480 comparisons per chunk. A partial selection algorithm or min-heap would reduce to O(n log k).
