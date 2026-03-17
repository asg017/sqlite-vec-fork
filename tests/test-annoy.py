import pytest
import struct
from helpers import _f32, exec


def test_annoy_create_basic(db):
    """Basic annoy index creation with defaults."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy()
        )"""
    )
    # Table should be created successfully
    tables = [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        ).fetchall()
    ]
    assert "t" in tables


def test_annoy_create_with_options(db):
    """Annoy with custom n_trees and search_k."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=100, search_k=500)
        )"""
    )
    tables = [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        ).fetchall()
    ]
    assert "t" in tables


def test_annoy_create_with_distance_metric(db):
    """Annoy with cosine distance metric."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] distance_metric=cosine INDEXED BY annoy(n_trees=10)
        )"""
    )
    tables = [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
        ).fetchall()
    ]
    assert "t" in tables


def test_annoy_create_error_bit_column(db):
    """Annoy should not work on bit columns."""
    result = exec(
        db,
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding bit[128] INDEXED BY annoy()
        )""",
    )
    assert "error" in result
    assert "bit" in result["message"].lower()


def test_annoy_create_error_bad_n_trees(db):
    """n_trees=0 should error."""
    result = exec(
        db,
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=0)
        )""",
    )
    assert "error" in result


def test_annoy_create_error_n_trees_too_large(db):
    """n_trees > 1000 should error."""
    result = exec(
        db,
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=9999)
        )""",
    )
    assert "error" in result


def test_annoy_create_error_unknown_option(db):
    """Unknown option should error."""
    result = exec(
        db,
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(bad_option=5)
        )""",
    )
    assert "error" in result


def test_annoy_create_error_missing_parens(db):
    """Missing parentheses should error."""
    result = exec(
        db,
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy
        )""",
    )
    assert "error" in result


def test_annoy_shadow_tables_created(db):
    """Annoy-indexed columns should create _annoy_nodes, _annoy_vectors, _annoy_buffer tables."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=10)
        )"""
    )
    tables = sorted(
        [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 't_%' ORDER BY 1"
            ).fetchall()
        ]
    )
    assert "t_annoy_buffer00" in tables
    assert "t_annoy_nodes00" in tables
    assert "t_annoy_vectors00" in tables
    # Should NOT have _vector_chunks00
    assert "t_vector_chunks00" not in tables


def test_annoy_info_seeded(db):
    """Annoy index should seed annoy_built in _info table."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy()
        )"""
    )
    rows = db.execute(
        "SELECT value FROM t_info WHERE key = 'annoy_built00'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 0


def test_annoy_non_annoy_no_extra_tables(db):
    """Non-annoy tables should not have annoy shadow tables."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4]
        )"""
    )
    tables = [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 't_annoy%'"
        ).fetchall()
    ]
    assert len(tables) == 0


def test_annoy_drop_table(db):
    """DROP TABLE should clean up annoy shadow tables."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy()
        )"""
    )
    db.execute("DROP TABLE t")
    tables = [
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 't_%'"
        ).fetchall()
    ]
    assert len(tables) == 0


def test_annoy_single_insert(db):
    """Single insert should write to _annoy_vectors and _annoy_buffer."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=2)
        )"""
    )
    db.execute(
        "INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])]
    )
    # Check _annoy_vectors has the row
    rows = db.execute("SELECT rowid FROM t_annoy_vectors00").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1

    # Check _annoy_buffer has the row (unindexed)
    rows = db.execute("SELECT rowid FROM t_annoy_buffer00").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1


def test_annoy_multiple_inserts(db):
    """Multiple inserts should all appear in vectors and buffer."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=2)
        )"""
    )
    for i in range(10):
        v = [0.0] * 4
        v[i % 4] = 1.0
        db.execute(
            "INSERT INTO t(rowid, embedding) VALUES (?, ?)", [i + 1, _f32(v)]
        )

    rows = db.execute("SELECT count(*) FROM t_annoy_vectors00").fetchone()
    assert rows[0] == 10

    rows = db.execute("SELECT count(*) FROM t_annoy_buffer00").fetchone()
    assert rows[0] == 10


def test_annoy_build_index(db):
    """build-index command should create tree nodes and clear buffer."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=3)
        )"""
    )
    for i in range(20):
        v = [float(i == j) for j in range(4)]
        db.execute(
            "INSERT INTO t(rowid, embedding) VALUES (?, ?)", [i + 1, _f32(v)]
        )

    # Build the index
    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    # Buffer should be empty now
    rows = db.execute("SELECT count(*) FROM t_annoy_buffer00").fetchone()
    assert rows[0] == 0

    # Nodes table should have entries
    rows = db.execute("SELECT count(*) FROM t_annoy_nodes00").fetchone()
    assert rows[0] > 0

    # Should have 3 distinct tree_ids
    rows = db.execute(
        "SELECT count(DISTINCT tree_id) FROM t_annoy_nodes00"
    ).fetchone()
    assert rows[0] == 3

    # annoy_built should be 1
    rows = db.execute(
        "SELECT value FROM t_info WHERE key = 'annoy_built00'"
    ).fetchone()
    assert rows[0] == 1


def test_annoy_rebuild_index(db):
    """rebuild-index should clear and rebuild trees."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=2)
        )"""
    )
    for i in range(10):
        v = [0.0] * 4
        v[i % 4] = 1.0
        db.execute(
            "INSERT INTO t(rowid, embedding) VALUES (?, ?)", [i + 1, _f32(v)]
        )

    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")
    nodes_before = db.execute("SELECT count(*) FROM t_annoy_nodes00").fetchone()[0]

    db.execute("INSERT INTO t(rowid, embedding) VALUES ('rebuild-index', NULL)")
    nodes_after = db.execute("SELECT count(*) FROM t_annoy_nodes00").fetchone()[0]

    # Should have roughly the same number of nodes
    assert nodes_after > 0
    assert nodes_before > 0


def test_annoy_delete(db):
    """Delete should remove from vectors table."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=2)
        )"""
    )
    db.execute(
        "INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])]
    )
    db.execute(
        "INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0])]
    )

    db.execute("DELETE FROM t WHERE rowid = 1")

    rows = db.execute("SELECT rowid FROM t_annoy_vectors00").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 2


def test_annoy_build_empty(db):
    """Building with no vectors should succeed."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=2)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    rows = db.execute(
        "SELECT value FROM t_info WHERE key = 'annoy_built00'"
    ).fetchone()
    assert rows[0] == 1


def test_annoy_knn_basic(db):
    """Basic KNN query should return the closest vector."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=5)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (3, ?)", [_f32([0, 0, 1, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (4, ?)", [_f32([0, 0, 0, 1])])

    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 1",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1] < 0.01  # Should be very close to 0


def test_annoy_knn_distances_sorted(db):
    """KNN results should be sorted by distance."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=10)
        )"""
    )
    # Insert vectors at known distances from query [1,0,0,0]
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0.9, 0.1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (3, ?)", [_f32([0.5, 0.5, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (4, ?)", [_f32([0, 1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (5, ?)", [_f32([0, 0, 1, 0])])

    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 5",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 5
    # Distances should be non-decreasing
    for i in range(len(rows) - 1):
        assert rows[i][1] <= rows[i + 1][1] + 0.001


def test_annoy_knn_empty_table(db):
    """KNN on empty table should return no results."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=2)
        )"""
    )
    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 5",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 0


def test_annoy_knn_before_build(db):
    """KNN before build should use buffer (brute-force)."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=2)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0])])

    # Query without building - should still work via buffer brute-force
    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 1",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1


def test_annoy_knn_after_delete(db):
    """KNN after delete should not return deleted items."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=3)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (3, ?)", [_f32([0, 0, 1, 0])])

    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")
    db.execute("DELETE FROM t WHERE rowid = 1")

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 3",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    # Should not contain rowid 1
    rowids = [r[0] for r in rows]
    assert 1 not in rowids
    assert len(rows) == 2


def test_annoy_knn_larger_dataset(db):
    """KNN on a larger dataset should have reasonable recall."""
    import random
    random.seed(42)

    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[8] INDEXED BY annoy(n_trees=10)
        )"""
    )

    n = 200
    vectors = []
    for i in range(n):
        v = [random.gauss(0, 1) for _ in range(8)]
        vectors.append(v)
        db.execute(
            "INSERT INTO t(rowid, embedding) VALUES (?, ?)", [i + 1, _f32(v)]
        )

    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    # Query with a known vector
    query = vectors[0]
    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 10",
        [_f32(query)],
    ).fetchall()
    assert len(rows) == 10
    # The exact match should be first (or very close)
    assert rows[0][0] == 1
    assert rows[0][1] < 0.01


def test_annoy_knn_cosine(db):
    """KNN with cosine distance should work correctly."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] distance_metric=cosine INDEXED BY annoy(n_trees=5)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (3, ?)", [_f32([0.9, 0.1, 0, 0])])

    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 3",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 3
    # rowid 1 should be closest (exact match), then 3 (similar direction)
    assert rows[0][0] == 1
    assert rows[0][1] < 0.01


def test_annoy_runtime_search_k(db):
    """Runtime search_k tuning via special insert command."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=5, search_k=10)
        )"""
    )
    for i in range(20):
        v = [0.0] * 4
        v[i % 4] = 1.0
        db.execute(
            "INSERT INTO t(rowid, embedding) VALUES (?, ?)", [i + 1, _f32(v)]
        )
    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    # Change search_k at runtime
    db.execute("INSERT INTO t(rowid, embedding) VALUES ('search_k=500', NULL)")

    # Query should still work
    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 5",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 5


def test_annoy_insert_after_build(db):
    """Inserts after build should go to buffer and be queryable."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] INDEXED BY annoy(n_trees=3)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    # Insert after build — goes to buffer
    db.execute("INSERT INTO t(rowid, embedding) VALUES (3, ?)", [_f32([0.99, 0.01, 0, 0])])

    # Buffer should have 1 item
    rows = db.execute("SELECT count(*) FROM t_annoy_buffer00").fetchone()
    assert rows[0] == 1

    # KNN should find the buffer item (it's very close to [1,0,0,0])
    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 3",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    rowids = [r[0] for r in rows]
    assert 3 in rowids  # buffer item should appear


def test_annoy_quantizer_int8(db):
    """int8 quantized split vectors should still produce correct results."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4] distance_metric=cosine
              INDEXED BY annoy(n_trees=5, quantizer=int8)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (3, ?)", [_f32([0.9, 0.1, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 3",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 3
    assert rows[0][0] == 1
    assert rows[0][1] < 0.01


def test_annoy_quantizer_binary(db):
    """Binary quantized split vectors should still produce correct results."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[8] distance_metric=cosine
              INDEXED BY annoy(n_trees=5, quantizer=binary)
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0, 0, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (2, ?)", [_f32([0, 1, 0, 0, 0, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES (3, ?)", [_f32([0.9, 0.1, 0, 0, 0, 0, 0, 0])])
    db.execute("INSERT INTO t(rowid, embedding) VALUES ('build-index', NULL)")

    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 3",
        [_f32([1, 0, 0, 0, 0, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 3
    assert rows[0][0] == 1


def test_annoy_no_index_still_works(db):
    """Non-annoy tables should still work fine."""
    db.execute(
        """CREATE VIRTUAL TABLE t USING vec0(
            embedding float[4]
        )"""
    )
    db.execute("INSERT INTO t(rowid, embedding) VALUES (1, ?)", [_f32([1, 0, 0, 0])])
    rows = db.execute(
        "SELECT rowid, distance FROM t WHERE embedding MATCH ? AND k = 1",
        [_f32([1, 0, 0, 0])],
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == 1
