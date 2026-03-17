# Arroy: Annoy-in-a-Database (LMDB)

Source: `/home/sprite/arroy/src/` (Rust, ~4000 lines)

## Key Insight

Arroy proves that annoy's tree structure can be stored in a transactional
key-value store instead of a flat mmap'd file, while preserving the algorithm's
performance characteristics. This is the closest prior art to what we want to
build in SQLite.

## Database Key Scheme

8-byte composite key:

```
[16 bits: index] [8 bits: mode] [32 bits: item_id] [8 bits: padding]
```

**Mode values:**
- `0 = Metadata` — single entry per index (dimensions, distance, roots, item bitmap)
- `1 = Updated` — tracks items modified since last build (dirty flags)
- `2 = Tree` — internal tree nodes (split planes, descendants)
- `3 = Item` — user-supplied leaf vectors

This enables efficient prefix scans: "give me all tree nodes for index X",
"give me all items for index X", etc.

## Node Serialization

Three node types stored as LMDB values:

### 1. Leaf (tag=0)
```
[1 byte: 0x00] [header bytes] [vector bytes]
```
Header is distance-metric-specific (e.g., norm for cosine).

### 2. Descendants (tag=1)
```
[1 byte: 0x01] [RoaringBitmap of item IDs]
```
Terminal tree node listing all items in this partition.
This replaces annoy's "small intermediate" optimization — instead of storing
a fixed array of IDs, uses a compressed bitmap.

### 3. SplitPlaneNormal (tag=2)
```
[1 byte: 0x02] [4 bytes: left_id] [4 bytes: right_id] [optional: split vector]
```
Internal node with hyperplane and two children.

## Build Process (writer.rs, ~1584 lines)

1. **Identify dirty items** via `Updated` keys (prefix scan)
2. **Delete removed items**: traverse existing trees, remove from descendants,
   collapse split nodes if children become empty
3. **Insert new items**: traverse frozen tree snapshots to find which
   descendants each new item belongs to, mark those descendants as needing splits
4. **Parallel tree building** with rayon:
   - Large descendants are subdivided recursively
   - New split nodes written to temporary files (TmpNodes)
   - Finally batch-committed to LMDB in one transaction
5. **Write metadata**: roots array, item bitmap, version info

### Transaction Strategy
- **Read snapshot** (RoTxn) frozen at build start for concurrent reads
- **Write transaction** (RwTxn) for final commit
- `Updated` keys act as a dirty flag — if any exist, reader refuses to query
  (returns `NeedBuild` error)

## Search Process (reader.rs)

Same priority queue algorithm as annoy:

```
queue = BinaryHeap with all roots
while candidates < search_k:
    pop (dist, node_id)
    match db.get(node_id):
        Leaf → compute distance, add to results
        Descendants → add all item IDs from bitmap to candidates
        SplitPlaneNormal → compute margin, push both children
```

Key difference from annoy: each node access is a database lookup (LMDB
b-tree traversal) instead of a direct array index. In practice LMDB's mmap
makes this nearly as fast for hot data.

## Incremental Updates

Unlike original annoy, arroy supports add/delete without full rebuild:

- **add_item()**: stores leaf in DB, marks as Updated
- **del_item()**: removes leaf, marks as Updated
- **build()**: processes only Updated items, modifying affected subtrees

This is crucial for sqlite-vec where we need INSERT/DELETE to work.

## Key Takeaways for SQLite Adaptation

1. **Node-per-row in database works**: arroy proves the model is viable
2. **RoaringBitmap for descendants**: more flexible than fixed arrays
3. **Dirty tracking enables incremental builds**: essential for SQL workflow
4. **Separate item storage from tree storage**: items keyed by mode=3,
   tree nodes by mode=2. In SQLite, these become separate shadow tables.
5. **Metadata row stores roots + dimensions + item count**: one row of config
6. **Build is a batch operation**: not per-insert. User triggers it explicitly.
   Between builds, new items are queryable via brute-force fallback or not at all.
