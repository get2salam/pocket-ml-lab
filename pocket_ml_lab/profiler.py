"""Schema inference and statistical profiling for tabular data."""

import math
from typing import Any

from .loader import column_names


def infer_schema(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Infer column types: 'numeric', 'categorical', or 'mixed'.

    A column is numeric if every non-None value is int or float.
    """
    if not rows:
        return {}
    schema: dict[str, str] = {}
    for col in column_names(rows):
        types = set()
        for row in rows:
            val = row.get(col)
            if val is None:
                continue
            if isinstance(val, (int, float)):
                types.add("numeric")
            else:
                types.add("categorical")
        if not types:
            schema[col] = "categorical"
        elif len(types) == 1:
            schema[col] = types.pop()
        else:
            schema[col] = "mixed"
    return schema


def profile_column(rows: list[dict[str, Any]], col: str) -> dict[str, Any]:
    """Compute summary statistics for one column."""
    values = [row[col] for row in rows if row.get(col) is not None]
    total = len(rows)
    null_count = total - len(values)
    result: dict[str, Any] = {
        "column": col,
        "total": total,
        "null_count": null_count,
        "null_pct": round(null_count / total * 100, 2) if total else 0.0,
    }

    numeric_vals = [v for v in values if isinstance(v, (int, float))]
    if numeric_vals:
        result["dtype"] = "numeric"
        result["count"] = len(numeric_vals)
        result["min"] = min(numeric_vals)
        result["max"] = max(numeric_vals)
        result["mean"] = round(sum(numeric_vals) / len(numeric_vals), 6)
        result["std"] = round(_std(numeric_vals), 6)
        result["median"] = _median(numeric_vals)
    else:
        result["dtype"] = "categorical"
        result["count"] = len(values)
        counts: dict[Any, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        result["cardinality"] = len(counts)
        result["top"] = sorted(counts, key=lambda k: -counts[k])[:5]
        result["top_counts"] = {str(k): counts[k] for k in result["top"]}

    return result


def profile_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Profile all columns and return a dataset-level summary."""
    schema = infer_schema(rows)
    columns = column_names(rows)
    profiles = {col: profile_column(rows, col) for col in columns}
    return {
        "n_rows": len(rows),
        "n_cols": len(columns),
        "columns": columns,
        "schema": schema,
        "profiles": profiles,
    }


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _median(values: list[float]) -> float:
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
