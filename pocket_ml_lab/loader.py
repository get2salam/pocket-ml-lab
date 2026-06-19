"""CSV loader with automatic type coercion and missing-value handling."""

import csv
import io
import os
from typing import Any


def load_csv(path: str) -> list[dict[str, Any]]:
    """Load a CSV file and return rows as a list of dicts.

    Values are coerced: integers first, then floats, then strings.
    Empty cells become None.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {path}")
        for raw_row in reader:
            row: dict[str, Any] = {}
            for key, value in raw_row.items():
                row[key] = _coerce(value)
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV is empty (no data rows): {path}")
    return rows


def load_csv_string(text: str) -> list[dict[str, Any]]:
    """Load CSV from a string — useful for tests and demos."""
    rows: list[dict[str, Any]] = []
    reader = csv.DictReader(io.StringIO(text))
    for raw_row in reader:
        rows.append({k: _coerce(v) for k, v in raw_row.items()})
    if not rows:
        raise ValueError("CSV string produced no data rows")
    return rows


def _coerce(value: str | None) -> Any:
    """Try int → float → str; map empty strings to None."""
    if value is None or value.strip() == "":
        return None
    stripped = value.strip()
    try:
        as_int = int(stripped)
        return as_int
    except ValueError:
        pass
    try:
        as_float = float(stripped)
        return as_float
    except ValueError:
        pass
    return stripped


def column_names(rows: list[dict[str, Any]]) -> list[str]:
    """Return column names from the first row."""
    if not rows:
        return []
    return list(rows[0].keys())
