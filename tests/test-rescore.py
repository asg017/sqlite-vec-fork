"""Tests for the rescore index feature in sqlite-vec."""
import struct
import sqlite3
import pytest
import math


@pytest.fixture()
def db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.enable_load_extension(True)
    db.load_extension("dist/vec0")
    db.enable_load_extension(False)
    return db


def float_vec(values):
    """Pack a list of floats into a blob for sqlite-vec."""
    return struct.pack(f"{len(values)}f", *values)


# ============================================================================
# Creation tests
# ============================================================================


def test_create_bit(db):
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "  embedding float[128] indexed by rescore(quantizer=bit)"
        ")"
    )
    # Table exists and has the right structure
    row = db.execute(
        "SELECT count(*) as cnt FROM sqlite_master WHERE name LIKE 't_%'"
    ).fetchone()
    assert row["cnt"] > 0


def test_create_int8(db):
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "  embedding float[128] indexed by rescore(quantizer=int8)"
        ")"
    )
    row = db.execute(
        "SELECT count(*) as cnt FROM sqlite_master WHERE name LIKE 't_%'"
    ).fetchone()
    assert row["cnt"] > 0


def test_create_with_oversample(db):
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "  embedding float[128] indexed by rescore(quantizer=bit, oversample=16)"
        ")"
    )
    row = db.execute(
        "SELECT count(*) as cnt FROM sqlite_master WHERE name LIKE 't_%'"
    ).fetchone()
    assert row["cnt"] > 0


def test_create_with_distance_metric(db):
    db.execute(
        "CREATE VIRTUAL TABLE t USING vec0("
        "  embedding float[128] distance_metric=cosine indexed by rescore(quantizer=bit)"
        ")"
    )
    row = db.execute(
        "SELECT count(*) as cnt FROM sqlite_master WHERE name LIKE 't_%'"
    ).fetchone()
    assert row["cnt"] > 0


def test_create_error_missing_quantizer(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  embedding float[128] indexed by rescore(oversample=8)"
            ")"
        )


def test_create_error_invalid_quantizer(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  embedding float[128] indexed by rescore(quantizer=float)"
            ")"
        )


def test_create_error_on_bit_column(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  embedding bit[1024] indexed by rescore(quantizer=bit)"
            ")"
        )


def test_create_error_on_int8_column(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  embedding int8[128] indexed by rescore(quantizer=bit)"
            ")"
        )


def test_create_error_bad_oversample_zero(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  embedding float[128] indexed by rescore(quantizer=bit, oversample=0)"
            ")"
        )


def test_create_error_bad_oversample_too_large(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  embedding float[128] indexed by rescore(quantizer=bit, oversample=999)"
            ")"
        )


def test_create_error_bit_dim_not_divisible_by_8(db):
    with pytest.raises(sqlite3.OperationalError):
        db.execute(
            "CREATE VIRTUAL TABLE t USING vec0("
            "  embedding float[100] indexed by rescore(quantizer=bit)"
            ")"
        )
