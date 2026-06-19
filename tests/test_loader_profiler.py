"""Tests for CSV loader and dataset profiler."""

import pytest

from pocket_ml_lab.loader import load_csv_string, column_names, _coerce
from pocket_ml_lab.profiler import infer_schema, profile_column, profile_dataset
from pocket_ml_lab.splitter import train_test_split


SIMPLE_CSV = """a,b,c
1,2.5,foo
2,3.0,bar
3,,baz
"""


def test_load_csv_string_basic():
    rows = load_csv_string(SIMPLE_CSV)
    assert len(rows) == 3
    assert rows[0]["a"] == 1
    assert rows[0]["b"] == 2.5
    assert rows[0]["c"] == "foo"


def test_load_csv_string_missing_value():
    rows = load_csv_string(SIMPLE_CSV)
    assert rows[2]["b"] is None


def test_column_names():
    rows = load_csv_string(SIMPLE_CSV)
    assert column_names(rows) == ["a", "b", "c"]


def test_coerce_int():
    assert _coerce("42") == 42


def test_coerce_float():
    assert _coerce("3.14") == 3.14


def test_coerce_string():
    assert _coerce("hello") == "hello"


def test_coerce_empty():
    assert _coerce("") is None
    assert _coerce("  ") is None


def test_infer_schema():
    rows = load_csv_string(SIMPLE_CSV)
    schema = infer_schema(rows)
    assert schema["a"] == "numeric"
    assert schema["b"] == "numeric"
    assert schema["c"] == "categorical"


def test_profile_numeric_column():
    rows = load_csv_string(SIMPLE_CSV)
    p = profile_column(rows, "a")
    assert p["dtype"] == "numeric"
    assert p["min"] == 1
    assert p["max"] == 3
    assert p["mean"] == 2.0
    assert p["null_count"] == 0


def test_profile_column_with_nulls():
    rows = load_csv_string(SIMPLE_CSV)
    p = profile_column(rows, "b")
    assert p["null_count"] == 1
    assert p["null_pct"] == pytest.approx(33.33, abs=0.1)


def test_profile_categorical_column():
    rows = load_csv_string(SIMPLE_CSV)
    p = profile_column(rows, "c")
    assert p["dtype"] == "categorical"
    assert p["cardinality"] == 3


def test_profile_dataset_keys():
    rows = load_csv_string(SIMPLE_CSV)
    ds = profile_dataset(rows)
    assert ds["n_rows"] == 3
    assert ds["n_cols"] == 3
    assert "profiles" in ds
    assert "schema" in ds


def test_split_sizes():
    rows = load_csv_string(SIMPLE_CSV + "\n4,1.1,qux\n5,2.2,zap")
    train, test = train_test_split(rows, test_size=0.2, seed=42)
    assert len(train) + len(test) == len(rows)
    assert len(test) >= 1


def test_split_determinism():
    rows = load_csv_string(SIMPLE_CSV + "\n4,1.1,qux\n5,2.2,zap\n6,3.3,woo")
    train1, test1 = train_test_split(rows, seed=42)
    train2, test2 = train_test_split(rows, seed=42)
    assert train1 == train2
    assert test1 == test2


def test_split_different_seeds():
    rows = load_csv_string(SIMPLE_CSV + "\n4,1.1,qux\n5,2.2,zap\n6,3.3,woo")
    _, test1 = train_test_split(rows, seed=1)
    _, test2 = train_test_split(rows, seed=2)
    assert test1 != test2
