// ============================================================
// Annoy vector I/O helpers
// ============================================================

/**
 * Write a vector to the _annoy_vectors shadow table.
 */
static int annoy_vector_write(vec0_vtab *p, int vec_col_idx, i64 rowid,
                               const void *vector, int vectorSize) {
  int rc;
  if (!p->stmtAnnoyVectorsInsert[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO " VEC0_SHADOW_ANNOY_VECTORS_N_NAME
        " (rowid, vector) VALUES (?, ?)",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyVectorsInsert[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyVectorsInsert[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  sqlite3_bind_blob(stmt, 2, vector, vectorSize, SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/**
 * Read a vector from the _annoy_vectors shadow table.
 * Returns SQLITE_OK and sets outVector (pointing into stmt memory) or error.
 */
static int annoy_vector_read(vec0_vtab *p, int vec_col_idx, i64 rowid,
                              const void **outVector, int *outSize) {
  int rc;
  if (!p->stmtAnnoyVectorsRead[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "SELECT vector FROM " VEC0_SHADOW_ANNOY_VECTORS_N_NAME
        " WHERE rowid = ?",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyVectorsRead[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyVectorsRead[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW) return SQLITE_ERROR;
  *outVector = sqlite3_column_blob(stmt, 0);
  *outSize = sqlite3_column_bytes(stmt, 0);
  return SQLITE_OK;
}

/**
 * Delete a vector from the _annoy_vectors shadow table.
 */
static int annoy_vector_delete(vec0_vtab *p, int vec_col_idx, i64 rowid) {
  int rc;
  if (!p->stmtAnnoyVectorsDelete[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "DELETE FROM " VEC0_SHADOW_ANNOY_VECTORS_N_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyVectorsDelete[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyVectorsDelete[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

// ============================================================
// Annoy buffer I/O helpers
// ============================================================

/**
 * Add a vector to the annoy buffer (unindexed items since last build).
 */
static int annoy_buffer_add(vec0_vtab *p, int vec_col_idx, i64 rowid,
                             const void *vector, int vectorSize) {
  int rc;
  if (!p->stmtAnnoyBufferInsert[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "INSERT OR REPLACE INTO " VEC0_SHADOW_ANNOY_BUFFER_N_NAME
        " (rowid, vector) VALUES (?, ?)",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyBufferInsert[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyBufferInsert[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  sqlite3_bind_blob(stmt, 2, vector, vectorSize, SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/**
 * Remove a vector from the annoy buffer.
 */
static int annoy_buffer_remove(vec0_vtab *p, int vec_col_idx, i64 rowid) {
  int rc;
  if (!p->stmtAnnoyBufferDelete[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "DELETE FROM " VEC0_SHADOW_ANNOY_BUFFER_N_NAME " WHERE rowid = ?",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyBufferDelete[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyBufferDelete[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int64(stmt, 1, rowid);
  rc = sqlite3_step(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/**
 * Count items in the annoy buffer.
 */
static int annoy_buffer_count(vec0_vtab *p, int vec_col_idx, i64 *outCount) {
  int rc;
  if (!p->stmtAnnoyBufferCount[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "SELECT count(*) FROM " VEC0_SHADOW_ANNOY_BUFFER_N_NAME,
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyBufferCount[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyBufferCount[vec_col_idx];
  sqlite3_reset(stmt);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW) return SQLITE_ERROR;
  *outCount = sqlite3_column_int64(stmt, 0);
  return SQLITE_OK;
}

// ============================================================
// Annoy node encode/decode
// ============================================================

/**
 * Encode a split node blob.
 * Format depends on quantizer:
 *   NONE:   [4B left] [4B right] [dims*4B float32 split_vector]
 *   INT8:   [4B left] [4B right] [dims*1B int8 split_vector]
 *   BINARY: [4B left] [4B right] [dims/8 B bit-packed split_vector]
 */
int annoy_encode_split_node(i32 left_id, i32 right_id,
                            const f32 *split_vector, int dimensions,
                            enum Vec0AnnoyQuantizerType quantizer,
                            u8 **outData, int *outSize) {
  int header = 2 * sizeof(i32);
  int vec_size;
  switch (quantizer) {
    case VEC0_ANNOY_QUANTIZER_INT8:
      vec_size = dimensions * sizeof(i8);
      break;
    case VEC0_ANNOY_QUANTIZER_BINARY:
      vec_size = (dimensions + 7) / 8;
      break;
    default: // NONE
      vec_size = dimensions * sizeof(f32);
      break;
  }
  int size = header + vec_size;
  u8 *data = sqlite3_malloc(size);
  if (!data) return SQLITE_NOMEM;
  memcpy(data, &left_id, sizeof(i32));
  memcpy(data + sizeof(i32), &right_id, sizeof(i32));

  u8 *vec_out = data + header;
  switch (quantizer) {
    case VEC0_ANNOY_QUANTIZER_INT8: {
      i8 *dst = (i8 *)vec_out;
      for (int i = 0; i < dimensions; i++) {
        f32 v = split_vector[i] * 127.0f;
        if (v > 127.0f) v = 127.0f;
        if (v < -127.0f) v = -127.0f;
        dst[i] = (i8)v;
      }
      break;
    }
    case VEC0_ANNOY_QUANTIZER_BINARY: {
      memset(vec_out, 0, vec_size);
      for (int i = 0; i < dimensions; i++) {
        if (split_vector[i] >= 0.0f) {
          vec_out[i / 8] |= (1 << (i % 8));
        }
      }
      break;
    }
    default:
      memcpy(vec_out, split_vector, vec_size);
      break;
  }

  *outData = data;
  *outSize = size;
  return SQLITE_OK;
}

/**
 * Decode a split node blob and compute the margin (dot product with query).
 * For quantized nodes, computes margin directly without full dequantization.
 */
int annoy_decode_split_node(const u8 *data, int dataSize, int dimensions,
                            enum Vec0AnnoyQuantizerType quantizer,
                            i32 *outLeft, i32 *outRight,
                            const f32 *queryVector, f32 *outMargin) {
  int header = 2 * sizeof(i32);
  if (dataSize < header) return SQLITE_ERROR;
  memcpy(outLeft, data, sizeof(i32));
  memcpy(outRight, data + sizeof(i32), sizeof(i32));

  const u8 *vec_data = data + header;

  switch (quantizer) {
    case VEC0_ANNOY_QUANTIZER_INT8: {
      // dot(query_float, split_int8) / 127.0
      const i8 *qvec = (const i8 *)vec_data;
      f32 dot = 0.0f;
      for (int i = 0; i < dimensions; i++) {
        dot += queryVector[i] * (f32)qvec[i];
      }
      *outMargin = dot / 127.0f;
      break;
    }
    case VEC0_ANNOY_QUANTIZER_BINARY: {
      // For each bit: if set, add query[i], else subtract query[i]
      // Equivalent to dot(query, sign(split_vector))
      f32 dot = 0.0f;
      for (int i = 0; i < dimensions; i++) {
        int bit = (vec_data[i / 8] >> (i % 8)) & 1;
        dot += bit ? queryVector[i] : -queryVector[i];
      }
      *outMargin = dot;
      break;
    }
    default: {
      // Float32: direct dot product
      const f32 *split = (const f32 *)vec_data;
      f32 dot = 0.0f;
      for (int i = 0; i < dimensions; i++) {
        dot += queryVector[i] * split[i];
      }
      *outMargin = dot;
      break;
    }
  }

  return SQLITE_OK;
}

/**
 * Encode a descendants (leaf bucket) node blob.
 * Format: [n * 8 bytes: packed i64 rowids]
 */
int annoy_encode_descendants_node(const i64 *rowids, int n_rowids,
                                   u8 **outData, int *outSize) {
  int size = n_rowids * sizeof(i64);
  if (size == 0) {
    *outData = NULL;
    *outSize = 0;
    return SQLITE_OK;
  }
  u8 *data = sqlite3_malloc(size);
  if (!data) return SQLITE_NOMEM;
  memcpy(data, rowids, size);
  *outData = data;
  *outSize = size;
  return SQLITE_OK;
}

/**
 * Decode a descendants node blob.
 * outRowids points into the data blob (not a copy).
 */
int annoy_decode_descendants_node(const u8 *data, int dataSize,
                                   const i64 **outRowids,
                                   int *outCount) {
  if (dataSize % sizeof(i64) != 0) return SQLITE_ERROR;
  *outRowids = (const i64 *)data;
  *outCount = dataSize / sizeof(i64);
  return SQLITE_OK;
}

// ============================================================
// Annoy node database I/O
// ============================================================

/**
 * Insert a node into the _annoy_nodes shadow table.
 */
static int annoy_node_insert(vec0_vtab *p, int vec_col_idx,
                              i32 node_id, int tree_id, int node_type,
                              const u8 *data, int dataSize) {
  int rc;
  if (!p->stmtAnnoyNodeInsert[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "INSERT INTO " VEC0_SHADOW_ANNOY_NODES_N_NAME
        " (node_id, tree_id, node_type, data) VALUES (?, ?, ?, ?)",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyNodeInsert[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyNodeInsert[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int(stmt, 1, node_id);
  sqlite3_bind_int(stmt, 2, tree_id);
  sqlite3_bind_int(stmt, 3, node_type);
  sqlite3_bind_blob(stmt, 4, data, dataSize, SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/**
 * Read a node from the _annoy_nodes shadow table.
 * Returns node_type and data blob pointer (from stmt memory).
 */
static int annoy_node_read(vec0_vtab *p, int vec_col_idx, i32 node_id,
                            int *outNodeType, const u8 **outData, int *outDataSize) {
  int rc;
  if (!p->stmtAnnoyNodeRead[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "SELECT node_type, data FROM " VEC0_SHADOW_ANNOY_NODES_N_NAME
        " WHERE node_id = ?",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyNodeRead[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyNodeRead[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int(stmt, 1, node_id);
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW) return SQLITE_ERROR;
  *outNodeType = sqlite3_column_int(stmt, 0);
  *outData = sqlite3_column_blob(stmt, 1);
  *outDataSize = sqlite3_column_bytes(stmt, 1);
  return SQLITE_OK;
}

/**
 * Delete all nodes for a given tree from the _annoy_nodes table.
 */
static int annoy_nodes_delete_tree(vec0_vtab *p, int vec_col_idx, int tree_id) {
  int rc;
  if (!p->stmtAnnoyNodeDeleteTree[vec_col_idx]) {
    char *zSql = sqlite3_mprintf(
        "DELETE FROM " VEC0_SHADOW_ANNOY_NODES_N_NAME " WHERE tree_id = ?",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) return SQLITE_NOMEM;
    rc = sqlite3_prepare_v2(p->db, zSql, -1,
                             &p->stmtAnnoyNodeDeleteTree[vec_col_idx], NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) return rc;
  }
  sqlite3_stmt *stmt = p->stmtAnnoyNodeDeleteTree[vec_col_idx];
  sqlite3_reset(stmt);
  sqlite3_bind_int(stmt, 1, tree_id);
  rc = sqlite3_step(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/**
 * Delete all nodes from the _annoy_nodes table (all trees).
 */
static int annoy_nodes_delete_all(vec0_vtab *p, int vec_col_idx) {
  sqlite3_stmt *stmt = NULL;
  char *zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_ANNOY_NODES_N_NAME,
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

// ============================================================
// Annoy info helpers (metadata in _info table)
// ============================================================

static int annoy_info_set_built(vec0_vtab *p, int vec_col_idx, int built) {
  char *key = sqlite3_mprintf("annoy_built%02d", vec_col_idx);
  if (!key) return SQLITE_NOMEM;
  char *zSql = sqlite3_mprintf(
      "UPDATE \"%w\".\"%w_info\" SET value = %d WHERE key = '%q'",
      p->schemaName, p->tableName, built, key);
  sqlite3_free(key);
  if (!zSql) return SQLITE_NOMEM;
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

static int annoy_info_get_built(vec0_vtab *p, int vec_col_idx, int *outBuilt) {
  char *key = sqlite3_mprintf("annoy_built%02d", vec_col_idx);
  if (!key) return SQLITE_NOMEM;
  char *zSql = sqlite3_mprintf(
      "SELECT value FROM \"%w\".\"%w_info\" WHERE key = '%q'",
      p->schemaName, p->tableName, key);
  sqlite3_free(key);
  if (!zSql) return SQLITE_NOMEM;
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW) {
    sqlite3_finalize(stmt);
    return SQLITE_ERROR;
  }
  *outBuilt = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return SQLITE_OK;
}

/**
 * Store annoy root node IDs in the _info table as a blob.
 */
static int annoy_info_set_roots(vec0_vtab *p, int vec_col_idx,
                                 const i32 *roots, int n_roots) {
  char *key = sqlite3_mprintf("annoy_roots%02d", vec_col_idx);
  if (!key) return SQLITE_NOMEM;

  // Upsert the roots blob
  char *zSql = sqlite3_mprintf(
      "INSERT OR REPLACE INTO \"%w\".\"%w_info\"(key, value) VALUES ('%q', ?)",
      p->schemaName, p->tableName, key);
  sqlite3_free(key);
  if (!zSql) return SQLITE_NOMEM;
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  sqlite3_bind_blob(stmt, 1, roots, n_roots * sizeof(i32), SQLITE_STATIC);
  rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

/**
 * Load annoy root node IDs from the _info table.
 * Caller must sqlite3_free(*outRoots).
 */
static int annoy_info_get_roots(vec0_vtab *p, int vec_col_idx,
                                 i32 **outRoots, int *outNRoots) {
  char *key = sqlite3_mprintf("annoy_roots%02d", vec_col_idx);
  if (!key) return SQLITE_NOMEM;
  char *zSql = sqlite3_mprintf(
      "SELECT value FROM \"%w\".\"%w_info\" WHERE key = '%q'",
      p->schemaName, p->tableName, key);
  sqlite3_free(key);
  if (!zSql) return SQLITE_NOMEM;
  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmt, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmt);
  if (rc != SQLITE_ROW || sqlite3_column_type(stmt, 0) == SQLITE_NULL) {
    sqlite3_finalize(stmt);
    *outRoots = NULL;
    *outNRoots = 0;
    return SQLITE_OK;
  }
  int blobSize = sqlite3_column_bytes(stmt, 0);
  const void *blob = sqlite3_column_blob(stmt, 0);
  int nRoots = blobSize / sizeof(i32);
  i32 *roots = sqlite3_malloc(blobSize);
  if (!roots) {
    sqlite3_finalize(stmt);
    return SQLITE_NOMEM;
  }
  memcpy(roots, blob, blobSize);
  sqlite3_finalize(stmt);
  *outRoots = roots;
  *outNRoots = nRoots;
  return SQLITE_OK;
}

// ============================================================
// Annoy two-means split
// ============================================================

/**
 * Compute dot product of two float vectors.
 */
static f32 annoy_dot(const f32 *a, const f32 *b, int dims) {
  f32 sum = 0.0f;
  for (int i = 0; i < dims; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * Normalize a float vector in-place.
 */
static void annoy_normalize(f32 *v, int dims) {
  f32 norm = 0.0f;
  for (int i = 0; i < dims; i++) {
    norm += v[i] * v[i];
  }
  norm = sqrtf(norm);
  if (norm > 0.0f) {
    for (int i = 0; i < dims; i++) {
      v[i] /= norm;
    }
  }
}

/**
 * Simple inline PRNG (xorshift32) for tree building.
 * Does not need to be cryptographically secure.
 */
static u32 annoy_rng_next(u32 *state) {
  u32 x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

/**
 * Cosine distance between two vectors: 2 - 2*cos(a,b)
 * Matches annoy's Angular::distance().
 */
static f32 annoy_cosine_distance(const f32 *a, const f32 *b, int dims) {
  f32 pp = 0, qq = 0, pq = 0;
  for (int d = 0; d < dims; d++) {
    pp += a[d] * a[d];
    qq += b[d] * b[d];
    pq += a[d] * b[d];
  }
  f32 denom = sqrtf(pp) * sqrtf(qq);
  if (denom > 0.0f)
    return 2.0f - 2.0f * pq / denom;
  return 2.0f;
}

/**
 * Two-means split following annoy's algorithm exactly.
 *
 * For cosine distance:
 * 1. Pick two random items as initial centroids, normalize them
 * 2. Run 200 iterations: assign random items to nearest centroid using
 *    cosine distance (2 - 2*cos), update centroid mean with normalized input
 * 3. Split vector = p - q (normalized)
 * 4. Partition items by sign of dot(item, split_vector)
 *
 * For L2/L1 distance:
 *    Same but use L2 distance and no normalization
 */
static void annoy_two_means_split(
    const f32 *vectors, const int *item_indices, int n_items, int dims,
    u32 *rng_state, int use_cosine,
    f32 *split_vector_out,
    int *left_out, int *n_left,
    int *right_out, int *n_right) {

  // Pick two random distinct points as initial centroids
  int pi = annoy_rng_next(rng_state) % n_items;
  int qi = (pi + 1 + annoy_rng_next(rng_state) % (n_items - 1)) % n_items;

  // Allocate centroids
  f32 *p = sqlite3_malloc(dims * sizeof(f32) * 2);
  f32 *q = p + dims;
  memcpy(p, vectors + (i64)item_indices[pi] * dims, dims * sizeof(f32));
  memcpy(q, vectors + (i64)item_indices[qi] * dims, dims * sizeof(f32));

  // For cosine/angular: normalize centroids (matches annoy)
  if (use_cosine) {
    annoy_normalize(p, dims);
    annoy_normalize(q, dims);
  }

  int ic = 1, jc = 1;

  // Two-means iterations (200 like annoy)
  for (int iter = 0; iter < 200; iter++) {
    int ki = annoy_rng_next(rng_state) % n_items;
    const f32 *k_vec = vectors + (i64)item_indices[ki] * dims;

    f32 dp, dq;
    if (use_cosine) {
      // Angular distance: 2 - 2*cos(p, k)
      dp = ic * annoy_cosine_distance(p, k_vec, dims);
      dq = jc * annoy_cosine_distance(q, k_vec, dims);
    } else {
      // L2 distance
      dp = 0.0f; dq = 0.0f;
      for (int d = 0; d < dims; d++) {
        f32 diffp = p[d] - k_vec[d];
        f32 diffq = q[d] - k_vec[d];
        dp += diffp * diffp;
        dq += diffq * diffq;
      }
      dp *= ic;
      dq *= jc;
    }

    if (dp < dq) {
      if (use_cosine) {
        // Normalize input before adding to centroid mean (matches annoy)
        f32 norm = 0.0f;
        for (int d = 0; d < dims; d++) norm += k_vec[d] * k_vec[d];
        norm = sqrtf(norm);
        if (norm > 0.0f) {
          for (int d = 0; d < dims; d++)
            p[d] = (p[d] * ic + k_vec[d] / norm) / (ic + 1);
        }
      } else {
        for (int d = 0; d < dims; d++)
          p[d] = (p[d] * ic + k_vec[d]) / (ic + 1);
      }
      ic++;
    } else if (dq < dp) {
      if (use_cosine) {
        f32 norm = 0.0f;
        for (int d = 0; d < dims; d++) norm += k_vec[d] * k_vec[d];
        norm = sqrtf(norm);
        if (norm > 0.0f) {
          for (int d = 0; d < dims; d++)
            q[d] = (q[d] * jc + k_vec[d] / norm) / (jc + 1);
        }
      } else {
        for (int d = 0; d < dims; d++)
          q[d] = (q[d] * jc + k_vec[d]) / (jc + 1);
      }
      jc++;
    }
    // If dp == dq, skip (matches annoy's else-if pattern)
  }

  // Split vector = p - q, normalized
  for (int d = 0; d < dims; d++) {
    split_vector_out[d] = p[d] - q[d];
  }
  annoy_normalize(split_vector_out, dims);

  sqlite3_free(p);

  // Partition items by dot product with split vector
  *n_left = 0;
  *n_right = 0;
  for (int i = 0; i < n_items; i++) {
    const f32 *v = vectors + (i64)item_indices[i] * dims;
    f32 margin = annoy_dot(v, split_vector_out, dims);
    if (margin < 0.0f) {
      left_out[(*n_left)++] = item_indices[i];
    } else {
      right_out[(*n_right)++] = item_indices[i];
    }
  }

  // Handle degenerate splits: retry with imbalance check (matches annoy)
  if (*n_left == 0 || *n_right == 0) {
    *n_left = 0;
    *n_right = 0;
    for (int i = 0; i < n_items; i++) {
      if (annoy_rng_next(rng_state) % 2 == 0) {
        left_out[(*n_left)++] = item_indices[i];
      } else {
        right_out[(*n_right)++] = item_indices[i];
      }
    }
    if (*n_left == 0) {
      left_out[0] = right_out[--(*n_right)];
      *n_left = 1;
    } else if (*n_right == 0) {
      right_out[0] = left_out[--(*n_left)];
      *n_right = 1;
    }
  }
}

// ============================================================
// Annoy recursive tree build
// ============================================================

/**
 * Recursively build one annoy tree, writing nodes to the shadow table.
 *
 * vectors: flat array of ALL float32 vectors
 * item_indices: indices of items in this subtree
 * n_items: count
 * dims: dimensionality
 * tree_id: which tree we're building
 * next_node_id: counter for assigning unique node IDs (incremented)
 * rng_state: random state
 * p: vtab for database writes
 * vec_col_idx: vector column index
 *
 * Returns the node_id of the root of this subtree, or -1 on error.
 */
static i32 annoy_build_tree_recursive(
    vec0_vtab *p, int vec_col_idx,
    const f32 *vectors, int *item_indices, int n_items, int dims,
    int tree_id, i32 *next_node_id, u32 *rng_state, int use_cosine,
    enum Vec0AnnoyQuantizerType quantizer) {

  i32 my_node_id = (*next_node_id)++;

  // Base case: few enough items, create a descendants node
  if (n_items <= VEC0_ANNOY_MAX_DESCENDANTS) {
    // Convert item indices to rowids
    // Note: item_indices are 0-based indices into the vectors array.
    // We need to store actual rowids. The caller maps indices to rowids.
    // Actually, for simplicity, we store the item_indices as-is here,
    // and the build_all function passes rowid arrays directly.
    u8 *data = NULL;
    int dataSize = 0;
    // item_indices contain rowids at this point (see annoy_build_all)
    i64 *rowids = sqlite3_malloc(n_items * sizeof(i64));
    if (!rowids) return -1;
    for (int i = 0; i < n_items; i++) {
      rowids[i] = (i64)item_indices[i];
    }
    int rc = annoy_encode_descendants_node(rowids, n_items, &data, &dataSize);
    sqlite3_free(rowids);
    if (rc != SQLITE_OK) return -1;
    rc = annoy_node_insert(p, vec_col_idx, my_node_id, tree_id,
                            VEC0_ANNOY_NODE_TYPE_DESCENDANTS, data, dataSize);
    sqlite3_free(data);
    if (rc != SQLITE_OK) return -1;
    return my_node_id;
  }

  // Recursive case: split and build subtrees
  f32 *split_vector = sqlite3_malloc(dims * sizeof(f32));
  int *left_indices = sqlite3_malloc(n_items * sizeof(int));
  int *right_indices = sqlite3_malloc(n_items * sizeof(int));
  if (!split_vector || !left_indices || !right_indices) {
    sqlite3_free(split_vector);
    sqlite3_free(left_indices);
    sqlite3_free(right_indices);
    return -1;
  }

  int n_left = 0, n_right = 0;
  annoy_two_means_split(vectors, item_indices, n_items, dims, rng_state,
                         use_cosine, split_vector, left_indices, &n_left,
                         right_indices, &n_right);

  // Build subtrees first to get their node IDs
  i32 left_child = annoy_build_tree_recursive(
      p, vec_col_idx, vectors, left_indices, n_left, dims,
      tree_id, next_node_id, rng_state, use_cosine, quantizer);
  i32 right_child = annoy_build_tree_recursive(
      p, vec_col_idx, vectors, right_indices, n_right, dims,
      tree_id, next_node_id, rng_state, use_cosine, quantizer);

  sqlite3_free(left_indices);
  sqlite3_free(right_indices);

  if (left_child < 0 || right_child < 0) {
    sqlite3_free(split_vector);
    return -1;
  }

  // Encode and write the split node (with optional quantization)
  u8 *data = NULL;
  int dataSize = 0;
  int rc = annoy_encode_split_node(left_child, right_child, split_vector, dims,
                                    quantizer, &data, &dataSize);
  sqlite3_free(split_vector);
  if (rc != SQLITE_OK) return -1;

  rc = annoy_node_insert(p, vec_col_idx, my_node_id, tree_id,
                          VEC0_ANNOY_NODE_TYPE_SPLIT, data, dataSize);
  sqlite3_free(data);
  if (rc != SQLITE_OK) return -1;

  return my_node_id;
}

// ============================================================
// Annoy build-index command
// ============================================================

/**
 * Build (or rebuild) all annoy trees for a given vector column.
 * Reads all vectors from _annoy_vectors, builds n_trees random trees,
 * stores nodes in _annoy_nodes, and updates metadata.
 */
static int annoy_build_all(vec0_vtab *p, int vec_col_idx) {
  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0AnnoyConfig *cfg = &col->annoy;
  int dims = (int)col->dimensions;
  int rc;

  // 1. Clear existing trees
  rc = annoy_nodes_delete_all(p, vec_col_idx);
  if (rc != SQLITE_OK) return rc;

  // Reset the prepared statement since we just deleted everything
  if (p->stmtAnnoyNodeInsert[vec_col_idx]) {
    sqlite3_finalize(p->stmtAnnoyNodeInsert[vec_col_idx]);
    p->stmtAnnoyNodeInsert[vec_col_idx] = NULL;
  }

  // 2. Read all vectors from _annoy_vectors
  char *zSql = sqlite3_mprintf(
      "SELECT rowid, vector FROM " VEC0_SHADOW_ANNOY_VECTORS_N_NAME
      " ORDER BY rowid",
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;

  sqlite3_stmt *stmtRead = NULL;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtRead, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;

  // Collect all vectors into a flat array
  int capacity = 1024;
  int n_items = 0;
  f32 *all_vectors = sqlite3_malloc(capacity * dims * sizeof(f32));
  int *all_indices = sqlite3_malloc(capacity * sizeof(int));  // stores rowids as ints
  if (!all_vectors || !all_indices) {
    sqlite3_free(all_vectors);
    sqlite3_free(all_indices);
    sqlite3_finalize(stmtRead);
    return SQLITE_NOMEM;
  }

  while ((rc = sqlite3_step(stmtRead)) == SQLITE_ROW) {
    if (n_items >= capacity) {
      capacity *= 2;
      f32 *newVecs = sqlite3_realloc(all_vectors, capacity * dims * sizeof(f32));
      int *newIdx = sqlite3_realloc(all_indices, capacity * sizeof(int));
      if (!newVecs || !newIdx) {
        sqlite3_free(all_vectors);
        sqlite3_free(all_indices);
        sqlite3_finalize(stmtRead);
        return SQLITE_NOMEM;
      }
      all_vectors = newVecs;
      all_indices = newIdx;
    }

    i64 rowid = sqlite3_column_int64(stmtRead, 0);
    const void *vec = sqlite3_column_blob(stmtRead, 1);
    int vecSize = sqlite3_column_bytes(stmtRead, 1);

    if (vecSize != dims * (int)sizeof(f32)) {
      sqlite3_free(all_vectors);
      sqlite3_free(all_indices);
      sqlite3_finalize(stmtRead);
      return SQLITE_ERROR;
    }

    memcpy(all_vectors + (i64)n_items * dims, vec, vecSize);
    all_indices[n_items] = (int)rowid;  // rowid as index for the split function
    n_items++;
  }
  sqlite3_finalize(stmtRead);

  if (n_items == 0) {
    // No vectors, just mark as built with no roots
    sqlite3_free(all_vectors);
    sqlite3_free(all_indices);
    rc = annoy_info_set_roots(p, vec_col_idx, NULL, 0);
    if (rc == SQLITE_OK) {
      rc = annoy_info_set_built(p, vec_col_idx, 1);
    }
    return rc;
  }

  // 3. Build n_trees random trees
  i32 *roots = sqlite3_malloc(cfg->n_trees * sizeof(i32));
  if (!roots) {
    sqlite3_free(all_vectors);
    sqlite3_free(all_indices);
    return SQLITE_NOMEM;
  }

  i32 next_node_id = 0;
  u32 rng_state = 42;  // deterministic seed for reproducibility
  int use_cosine = (col->distance_metric == VEC0_DISTANCE_METRIC_COSINE);

  for (int t = 0; t < cfg->n_trees; t++) {
    // Each tree gets a copy of all_indices since the split function
    // doesn't modify the original, but we need the original for each tree
    int *tree_indices = sqlite3_malloc(n_items * sizeof(int));
    if (!tree_indices) {
      sqlite3_free(roots);
      sqlite3_free(all_vectors);
      sqlite3_free(all_indices);
      return SQLITE_NOMEM;
    }
    memcpy(tree_indices, all_indices, n_items * sizeof(int));

    i32 root = annoy_build_tree_recursive(
        p, vec_col_idx, all_vectors, tree_indices, n_items, dims,
        t, &next_node_id, &rng_state, use_cosine, cfg->quantizer);
    sqlite3_free(tree_indices);

    if (root < 0) {
      sqlite3_free(roots);
      sqlite3_free(all_vectors);
      sqlite3_free(all_indices);
      return SQLITE_ERROR;
    }
    roots[t] = root;
  }

  sqlite3_free(all_vectors);
  sqlite3_free(all_indices);

  // 4. Store roots and mark as built
  rc = annoy_info_set_roots(p, vec_col_idx, roots, cfg->n_trees);
  sqlite3_free(roots);
  if (rc != SQLITE_OK) return rc;

  rc = annoy_info_set_built(p, vec_col_idx, 1);
  if (rc != SQLITE_OK) return rc;

  // 5. Clear the buffer (all items are now indexed)
  zSql = sqlite3_mprintf(
      "DELETE FROM " VEC0_SHADOW_ANNOY_BUFFER_N_NAME,
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) return SQLITE_NOMEM;
  sqlite3_stmt *stmtClear = NULL;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtClear, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) return rc;
  rc = sqlite3_step(stmtClear);
  sqlite3_finalize(stmtClear);

  return rc == SQLITE_DONE ? SQLITE_OK : rc;
}

// ============================================================
// Annoy search (priority queue traversal across all trees)
// ============================================================

/**
 * Priority queue entry for annoy search.
 */
struct AnnoyPQEntry {
  f32 distance;  // distance estimate (lower = explore first)
  i32 node_id;
};

/**
 * Simple max-heap priority queue for annoy search.
 * We use a min-heap by negating distances (explore closest first).
 */
struct AnnoyPQ {
  struct AnnoyPQEntry *entries;
  int size;
  int capacity;
};

static int annoy_pq_init(struct AnnoyPQ *pq, int capacity) {
  pq->entries = sqlite3_malloc(capacity * sizeof(struct AnnoyPQEntry));
  if (!pq->entries) return SQLITE_NOMEM;
  pq->size = 0;
  pq->capacity = capacity;
  return SQLITE_OK;
}

static void annoy_pq_free(struct AnnoyPQ *pq) {
  sqlite3_free(pq->entries);
  pq->entries = NULL;
  pq->size = 0;
}

static void annoy_pq_swap(struct AnnoyPQEntry *a, struct AnnoyPQEntry *b) {
  struct AnnoyPQEntry tmp = *a;
  *a = *b;
  *b = tmp;
}

// Max-heap: largest priority value at top (matching annoy's convention
// where higher pq_distance = more promising to explore first)
static void annoy_pq_push(struct AnnoyPQ *pq, f32 distance, i32 node_id) {
  if (pq->size >= pq->capacity) {
    int newCap = pq->capacity * 2;
    struct AnnoyPQEntry *newEntries = sqlite3_realloc(pq->entries,
        newCap * sizeof(struct AnnoyPQEntry));
    if (!newEntries) return;
    pq->entries = newEntries;
    pq->capacity = newCap;
  }
  int i = pq->size++;
  pq->entries[i].distance = distance;
  pq->entries[i].node_id = node_id;
  // Bubble up (max-heap: parent >= children)
  while (i > 0) {
    int parent = (i - 1) / 2;
    if (pq->entries[parent].distance < pq->entries[i].distance) {
      annoy_pq_swap(&pq->entries[parent], &pq->entries[i]);
      i = parent;
    } else {
      break;
    }
  }
}

static struct AnnoyPQEntry annoy_pq_pop(struct AnnoyPQ *pq) {
  struct AnnoyPQEntry top = pq->entries[0];
  pq->entries[0] = pq->entries[--pq->size];
  // Bubble down (max-heap)
  int i = 0;
  while (1) {
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    int largest = i;
    if (left < pq->size && pq->entries[left].distance > pq->entries[largest].distance)
      largest = left;
    if (right < pq->size && pq->entries[right].distance > pq->entries[largest].distance)
      largest = right;
    if (largest != i) {
      annoy_pq_swap(&pq->entries[i], &pq->entries[largest]);
      i = largest;
    } else {
      break;
    }
  }
  return top;
}

/**
 * Compare function for sorting (rowid, distance) pairs by distance.
 */
struct AnnoyCandidate {
  i64 rowid;
  f32 distance;
};

static int annoy_candidate_cmp(const void *a, const void *b) {
  f32 da = ((const struct AnnoyCandidate *)a)->distance;
  f32 db = ((const struct AnnoyCandidate *)b)->distance;
  if (da < db) return -1;
  if (da > db) return 1;
  return 0;
}

static int annoy_cmp_i64(const void *a, const void *b) {
  i64 va = *(const i64 *)a;
  i64 vb = *(const i64 *)b;
  return (va > vb) - (va < vb);
}

/**
 * Main annoy search function.
 *
 * Traverses all trees simultaneously via priority queue, collects candidate
 * item rowids, then computes exact distances and returns top-k.
 */
static int annoy_search(
    vec0_vtab *p, int vec_col_idx,
    const void *queryVector, int k,
    i64 *outRowids, f32 *outDistances, int *outCount) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  struct Vec0AnnoyConfig *cfg = &col->annoy;
  int dims = (int)col->dimensions;
  int rc;

  // Determine search_k
  int search_k = cfg->search_k;
  if (search_k <= 0) {
    // Auto search_k: original annoy uses k * n_trees. We use 10x that
    // for high recall in high dimensions (768-dim cosine). Users can
    // override via search_k parameter or runtime 'search_k=N' command.
    search_k = k * cfg->n_trees * 10;
  }
  if (search_k < k) search_k = k;

  // Load roots
  i32 *roots = NULL;
  int n_roots = 0;
  rc = annoy_info_get_roots(p, vec_col_idx, &roots, &n_roots);
  if (rc != SQLITE_OK) return rc;

  if (n_roots == 0 || !roots) {
    // No index built yet
    *outCount = 0;
    sqlite3_free(roots);
    return SQLITE_OK;
  }

  // Initialize priority queue
  struct AnnoyPQ pq;
  rc = annoy_pq_init(&pq, search_k * 2);
  if (rc != SQLITE_OK) {
    sqlite3_free(roots);
    return rc;
  }

  // Push all roots with +infinity (max-heap pops largest first)
  for (int i = 0; i < n_roots; i++) {
    annoy_pq_push(&pq, FLT_MAX, roots[i]);
  }
  sqlite3_free(roots);

  // Collect candidate rowids
  int cand_capacity = search_k * 2;
  int n_candidates = 0;
  i64 *candidates = sqlite3_malloc(cand_capacity * sizeof(i64));
  if (!candidates) {
    annoy_pq_free(&pq);
    return SQLITE_NOMEM;
  }

  // Batch PQ traversal: pop a batch of nodes, read them in one query, process
  #define ANNOY_NODE_BATCH_SIZE 64
  struct AnnoyPQEntry node_batch[ANNOY_NODE_BATCH_SIZE];

  while (pq.size > 0 && n_candidates < search_k) {
    // Pop a batch of entries from the PQ
    int batch_n = 0;
    while (pq.size > 0 && batch_n < ANNOY_NODE_BATCH_SIZE && n_candidates + batch_n * VEC0_ANNOY_MAX_DESCENDANTS < search_k * 2) {
      node_batch[batch_n++] = annoy_pq_pop(&pq);
    }
    if (batch_n == 0) break;

    // Build JSON array of node IDs for batch SELECT via json_each
    sqlite3_str *json = sqlite3_str_new(NULL);
    sqlite3_str_appendall(json, "[");
    for (int j = 0; j < batch_n; j++) {
      if (j > 0) sqlite3_str_appendall(json, ",");
      sqlite3_str_appendf(json, "%d", node_batch[j].node_id);
    }
    sqlite3_str_appendall(json, "]");
    char *json_ids = sqlite3_str_finish(json);
    if (!json_ids) break;

    // Use json_each to avoid dynamic SQL: bind the JSON array as a parameter
    char *zSql = sqlite3_mprintf(
        "SELECT node_id, node_type, data FROM " VEC0_SHADOW_ANNOY_NODES_N_NAME
        " WHERE node_id IN (SELECT value FROM json_each(?))",
        p->schemaName, p->tableName, vec_col_idx);
    if (!zSql) { sqlite3_free(json_ids); break; }

    sqlite3_stmt *stmtBatch = NULL;
    rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtBatch, NULL);
    sqlite3_free(zSql);
    if (rc != SQLITE_OK) { sqlite3_free(json_ids); break; }
    sqlite3_bind_text(stmtBatch, 1, json_ids, -1, sqlite3_free);

    // Build a map from node_id -> batch index to recover PQ priority
    // (SQLite may return rows in any order)
    while (sqlite3_step(stmtBatch) == SQLITE_ROW) {
      i32 node_id = sqlite3_column_int(stmtBatch, 0);
      int node_type = sqlite3_column_int(stmtBatch, 1);
      const u8 *data = sqlite3_column_blob(stmtBatch, 2);
      int dataSize = sqlite3_column_bytes(stmtBatch, 2);

      // Find the matching PQ entry to get the priority
      f32 entry_distance = FLT_MAX;
      for (int j = 0; j < batch_n; j++) {
        if (node_batch[j].node_id == node_id) {
          entry_distance = node_batch[j].distance;
          break;
        }
      }

      if (node_type == VEC0_ANNOY_NODE_TYPE_DESCENDANTS) {
        const i64 *rowids;
        int count;
        rc = annoy_decode_descendants_node(data, dataSize, &rowids, &count);
        if (rc == SQLITE_OK) {
          for (int i = 0; i < count; i++) {
            if (n_candidates >= cand_capacity) {
              cand_capacity *= 2;
              i64 *newCands = sqlite3_realloc(candidates, cand_capacity * sizeof(i64));
              if (!newCands) break;
              candidates = newCands;
            }
            candidates[n_candidates++] = rowids[i];
          }
        }
      } else if (node_type == VEC0_ANNOY_NODE_TYPE_SPLIT) {
        i32 left, right;
        f32 margin;
        rc = annoy_decode_split_node(data, dataSize, dims,
                                      col->annoy.quantizer,
                                      &left, &right,
                                      (const f32 *)queryVector, &margin);
        if (rc == SQLITE_OK) {
          f32 d = entry_distance;
          f32 left_prio  = d < -margin ? d : -margin;
          f32 right_prio = d <  margin ? d :  margin;
          annoy_pq_push(&pq, left_prio, left);
          annoy_pq_push(&pq, right_prio, right);
        }
      }
    }
    sqlite3_finalize(stmtBatch);
  }
  annoy_pq_free(&pq);

  // Deduplicate candidates by sorting by rowid
  qsort(candidates, n_candidates, sizeof(i64), annoy_cmp_i64);

  // Remove duplicates in-place (candidates are i64 rowids)
  int unique = 0;
  for (int i = 0; i < n_candidates; i++) {
    if (unique == 0 || candidates[i] != candidates[unique - 1]) {
      candidates[unique++] = candidates[i];
    }
  }
  n_candidates = unique;

  // Compute exact distances for all candidates using batch vector read
  struct AnnoyCandidate *results = sqlite3_malloc(n_candidates * sizeof(struct AnnoyCandidate));
  if (!results) {
    sqlite3_free(candidates);
    return SQLITE_NOMEM;
  }

  int n_results = 0;

  // Batch vector reads via json_each for re-ranking
  // Process in chunks to keep JSON arrays reasonable
  {
    int batch_size = 500;
    for (int batch_start = 0; batch_start < n_candidates; batch_start += batch_size) {
      int batch_end = batch_start + batch_size;
      if (batch_end > n_candidates) batch_end = n_candidates;
      int batch_n = batch_end - batch_start;

      // Build JSON array of rowids
      sqlite3_str *json = sqlite3_str_new(NULL);
      sqlite3_str_appendall(json, "[");
      for (int j = 0; j < batch_n; j++) {
        if (j > 0) sqlite3_str_appendall(json, ",");
        sqlite3_str_appendf(json, "%lld", candidates[batch_start + j]);
      }
      sqlite3_str_appendall(json, "]");
      char *json_ids = sqlite3_str_finish(json);
      if (!json_ids) continue;

      char *zSql = sqlite3_mprintf(
          "SELECT rowid, vector FROM " VEC0_SHADOW_ANNOY_VECTORS_N_NAME
          " WHERE rowid IN (SELECT value FROM json_each(?))",
          p->schemaName, p->tableName, vec_col_idx);
      if (!zSql) { sqlite3_free(json_ids); continue; }

      sqlite3_stmt *stmtBatch = NULL;
      rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtBatch, NULL);
      sqlite3_free(zSql);
      if (rc != SQLITE_OK) { sqlite3_free(json_ids); continue; }
      sqlite3_bind_text(stmtBatch, 1, json_ids, -1, sqlite3_free);

      while (sqlite3_step(stmtBatch) == SQLITE_ROW) {
        i64 rowid = sqlite3_column_int64(stmtBatch, 0);
        const void *vec = sqlite3_column_blob(stmtBatch, 1);
        if (!vec) continue;

        f32 dist = vec0_distance_full(queryVector, vec, dims,
                                          col->element_type, col->distance_metric);
        results[n_results].rowid = rowid;
        results[n_results].distance = dist;
        n_results++;
      }
      sqlite3_finalize(stmtBatch);
    }
  }
  sqlite3_free(candidates);

  // Sort by distance and take top k
  qsort(results, n_results, sizeof(struct AnnoyCandidate), annoy_candidate_cmp);

  int out_n = n_results < k ? n_results : k;
  for (int i = 0; i < out_n; i++) {
    outRowids[i] = results[i].rowid;
    outDistances[i] = results[i].distance;
  }
  *outCount = out_n;

  sqlite3_free(results);
  return SQLITE_OK;
}

/**
 * Also search the buffer (brute-force) and merge with tree results.
 */
static int annoy_search_with_buffer(
    vec0_vtab *p, int vec_col_idx,
    const void *queryVector, int k,
    i64 *outRowids, f32 *outDistances, int *outCount) {

  struct VectorColumnDefinition *col = &p->vector_columns[vec_col_idx];
  int dims = (int)col->dimensions;
  int rc;

  // First do tree search
  int tree_count = 0;
  i64 *tree_rowids = sqlite3_malloc(k * sizeof(i64));
  f32 *tree_dists = sqlite3_malloc(k * sizeof(f32));
  if (!tree_rowids || !tree_dists) {
    sqlite3_free(tree_rowids);
    sqlite3_free(tree_dists);
    return SQLITE_NOMEM;
  }

  int built = 0;
  annoy_info_get_built(p, vec_col_idx, &built);
  if (built) {
    rc = annoy_search(p, vec_col_idx, queryVector, k,
                       tree_rowids, tree_dists, &tree_count);
    if (rc != SQLITE_OK) {
      sqlite3_free(tree_rowids);
      sqlite3_free(tree_dists);
      return rc;
    }
  }

  // Now scan buffer
  char *zSql = sqlite3_mprintf(
      "SELECT rowid, vector FROM " VEC0_SHADOW_ANNOY_BUFFER_N_NAME,
      p->schemaName, p->tableName, vec_col_idx);
  if (!zSql) {
    sqlite3_free(tree_rowids);
    sqlite3_free(tree_dists);
    return SQLITE_NOMEM;
  }
  sqlite3_stmt *stmtBuf = NULL;
  rc = sqlite3_prepare_v2(p->db, zSql, -1, &stmtBuf, NULL);
  sqlite3_free(zSql);
  if (rc != SQLITE_OK) {
    sqlite3_free(tree_rowids);
    sqlite3_free(tree_dists);
    return rc;
  }

  // Merge buffer results with tree results
  int merge_cap = tree_count + 256;
  int merge_count = 0;
  struct AnnoyCandidate *merged = sqlite3_malloc(merge_cap * sizeof(struct AnnoyCandidate));
  if (!merged) {
    sqlite3_finalize(stmtBuf);
    sqlite3_free(tree_rowids);
    sqlite3_free(tree_dists);
    return SQLITE_NOMEM;
  }

  // Add tree results
  for (int i = 0; i < tree_count; i++) {
    merged[merge_count].rowid = tree_rowids[i];
    merged[merge_count].distance = tree_dists[i];
    merge_count++;
  }
  sqlite3_free(tree_rowids);
  sqlite3_free(tree_dists);

  // Add buffer results
  while ((rc = sqlite3_step(stmtBuf)) == SQLITE_ROW) {
    if (merge_count >= merge_cap) {
      merge_cap *= 2;
      struct AnnoyCandidate *newMerged = sqlite3_realloc(merged,
          merge_cap * sizeof(struct AnnoyCandidate));
      if (!newMerged) break;
      merged = newMerged;
    }
    i64 rowid = sqlite3_column_int64(stmtBuf, 0);
    const void *vec = sqlite3_column_blob(stmtBuf, 1);
    f32 dist = vec0_distance_full(queryVector, vec, dims,
                                      col->element_type, col->distance_metric);
    merged[merge_count].rowid = rowid;
    merged[merge_count].distance = dist;
    merge_count++;
  }
  sqlite3_finalize(stmtBuf);

  // Sort and take top k
  qsort(merged, merge_count, sizeof(struct AnnoyCandidate), annoy_candidate_cmp);
  int out_n = merge_count < k ? merge_count : k;
  for (int i = 0; i < out_n; i++) {
    outRowids[i] = merged[i].rowid;
    outDistances[i] = merged[i].distance;
  }
  *outCount = out_n;
  sqlite3_free(merged);

  return SQLITE_OK;
}
