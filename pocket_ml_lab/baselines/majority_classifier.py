"""Majority-class classifier — predicts the most frequent training class."""

from typing import Any


class MajorityClassifier:
    """Baseline classifier that always predicts the majority class.

    This is the minimum bar any real classifier must beat.
    """

    name = "MajorityClassifier"

    def __init__(self) -> None:
        self._majority_class: Any = None
        self._class_counts: dict[Any, int] = {}

    def fit(self, X: list[dict[str, Any]], y: list[Any]) -> "MajorityClassifier":
        if not y:
            raise ValueError("Cannot fit on empty label list")
        self._class_counts = {}
        for label in y:
            self._class_counts[label] = self._class_counts.get(label, 0) + 1
        self._majority_class = max(self._class_counts, key=lambda k: self._class_counts[k])
        return self

    def predict(self, X: list[dict[str, Any]]) -> list[Any]:
        if self._majority_class is None:
            raise RuntimeError("Call fit() before predict()")
        return [self._majority_class] * len(X)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "majority_class": self._majority_class,
            "class_counts": self._class_counts,
        }
