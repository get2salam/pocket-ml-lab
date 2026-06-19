"""Tests for classification and regression metrics."""

import math
import pytest

from pocket_ml_lab.metrics.classification import (
    accuracy,
    confusion_matrix,
    per_class_metrics,
    classification_report,
)
from pocket_ml_lab.metrics.regression import mae, rmse, r2_score


# ── Classification ────────────────────────────────────────────────────────────

def test_accuracy_perfect():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_all_wrong():
    assert accuracy([0, 0, 0], [1, 1, 1]) == 0.0


def test_accuracy_partial():
    assert accuracy([0, 1, 1, 0], [0, 0, 1, 0]) == pytest.approx(0.75)


def test_accuracy_empty():
    assert accuracy([], []) == 0.0


def test_confusion_matrix_binary():
    classes, matrix = confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1])
    assert classes == [0, 1]
    assert matrix == [[1, 1], [1, 1]]


def test_confusion_matrix_perfect():
    classes, matrix = confusion_matrix(["a", "b", "c"], ["a", "b", "c"])
    assert classes == ["a", "b", "c"]
    for i in range(3):
        assert matrix[i][i] == 1
        for j in range(3):
            if i != j:
                assert matrix[i][j] == 0


def test_per_class_precision_recall():
    # TP=2, FP=0, FN=0 for class 'a'; TP=1, FP=0, FN=0 for class 'b'
    y_true = ["a", "a", "b"]
    y_pred = ["a", "a", "b"]
    pcm = per_class_metrics(y_true, y_pred)
    assert pcm["a"]["precision"] == pytest.approx(1.0)
    assert pcm["a"]["recall"] == pytest.approx(1.0)
    assert pcm["a"]["f1"] == pytest.approx(1.0)


def test_per_class_zero_precision():
    y_true = ["a", "a"]
    y_pred = ["b", "b"]
    pcm = per_class_metrics(y_true, y_pred)
    assert pcm["a"]["precision"] == 0.0
    assert pcm["b"]["recall"] == 0.0


def test_classification_report_keys():
    report = classification_report(["a", "b", "a"], ["a", "a", "a"])
    assert "accuracy" in report
    assert "macro_f1" in report
    assert "per_class" in report
    assert "confusion_matrix" in report


def test_classification_report_length_mismatch():
    with pytest.raises(ValueError):
        classification_report([0, 1], [0])


# ── Regression ────────────────────────────────────────────────────────────────

def test_mae_perfect():
    assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_constant_error():
    assert mae([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_rmse_perfect():
    assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_rmse_known_value():
    # errors = [1, 1, 1] → RMSE = 1
    assert rmse([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]) == pytest.approx(1.0)


def test_r2_perfect():
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r2_mean_predictor():
    # Predicting the mean always gives R² = 0
    y = [1.0, 2.0, 3.0]
    mean = sum(y) / len(y)
    assert r2_score(y, [mean, mean, mean]) == pytest.approx(0.0, abs=1e-9)


def test_r2_negative():
    # Predicting all zeros for [1, 2, 3] gives R² < 0
    assert r2_score([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) < 0


def test_regression_length_mismatch():
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])
