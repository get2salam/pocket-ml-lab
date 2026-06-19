"""JSON report writer — serialises experiment results to a structured JSON file."""

import json
import os
from typing import Any


def format_json(results: dict[str, Any]) -> str:
    """Render experiment results as a pretty-printed JSON string."""
    return json.dumps(results, indent=2, default=_serialise)


def write_json_report(results: dict[str, Any], path: str) -> str:
    """Write experiment results to *path* as JSON and return the path."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    text = format_json(results)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def _serialise(obj: Any) -> Any:
    """Fallback serialiser for non-JSON-native types."""
    if isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
