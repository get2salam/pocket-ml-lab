"""Classification metrics: accuracy, precision, recall, F1, confusion matrix."""

from typing import Any


def accuracy(y_true: list[Any], y_pred: list[Any]) -> float:
    """Fraction of correct predictions."""
    _check_lengths(y_true, y_pred)
    if not y_true:
        return 0.0
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true)


def confusion_matrix(
    y_true: list[Any], y_pred: list[Any]
) -> tuple[list[Any], list[list[int]]]:
    """Return (classes, matrix) where matrix[i][j] is count of true=i predicted=j."""
    _check_lengths(y_true, y_pred)
    classes = sorted(set(y_true) | set(y_pred), key=str)
    idx = {cls: i for i, cls in enumerate(classes)}
    n = len(classes)
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    return classes, matrix


def per_class_metrics(
    y_true: list[Any], y_pred: list[Any]
) -> dict[Any, dict[str, float]]:
    """Per-class precision, recall, and F1 scores."""
    classes, matrix = confusion_matrix(y_true, y_pred)
    n = len(classes)
    result = {}
    for i, cls in enumerate(classes):
        tp = matrix[i][i]
        fp = sum(matrix[j][i] for j in range(n)) - tp
        fn = sum(matrix[i][j] for j in range(n)) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        result[cls] = {
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "f1": round(f1, 6),
            "support": sum(matrix[i]),
        }
    return result


def balanced_accuracy(y_true: list[Any], y_pred: list[Any]) -> float:
    """Mean recall across true classes for imbalanced classification evaluation.

    Unlike plain accuracy, balanced accuracy gives each observed class the same
    weight. This makes majority-class baselines easier to audit when a small
    class is missed entirely.
    """
    _check_lengths(y_true, y_pred)
    if not y_true:
        return 0.0

    recalls = []
    for cls in sorted(set(y_true), key=str):
        support = sum(t == cls for t in y_true)
        true_positive = sum(t == cls and p == cls for t, p in zip(y_true, y_pred))
        recalls.append(true_positive / support if support else 0.0)
    return sum(recalls) / len(recalls)


def classification_report(
    y_true: list[Any], y_pred: list[Any]
) -> dict[str, Any]:
    """Full classification report: accuracy, per-class metrics, macro averages."""
    pcm = per_class_metrics(y_true, y_pred)
    classes = list(pcm)
    macro_prec = sum(pcm[c]["precision"] for c in classes) / len(classes) if classes else 0.0
    macro_rec = sum(pcm[c]["recall"] for c in classes) / len(classes) if classes else 0.0
    macro_f1 = sum(pcm[c]["f1"] for c in classes) / len(classes) if classes else 0.0
    classes_cm, matrix = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "macro_precision": round(macro_prec, 6),
        "macro_recall": round(macro_rec, 6),
        "macro_f1": round(macro_f1, 6),
        "balanced_accuracy": round(balanced_accuracy(y_true, y_pred), 6),
        "per_class": pcm,
        "confusion_matrix": {
            "classes": [str(c) for c in classes_cm],
            "matrix": matrix,
        },
    }


def _check_lengths(a: list, b: list) -> None:
    if len(a) != len(b):
        raise ValueError(f"y_true and y_pred must have the same length ({len(a)} vs {len(b)})")
