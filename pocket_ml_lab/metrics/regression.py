"""Regression metrics: MAE, RMSE, and R²."""

import math


def mae(y_true: list[float], y_pred: list[float]) -> float:
    """Mean Absolute Error."""
    _check(y_true, y_pred)
    if not y_true:
        return 0.0
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def rmse(y_true: list[float], y_pred: list[float]) -> float:
    """Root Mean Squared Error."""
    _check(y_true, y_pred)
    if not y_true:
        return 0.0
    mse = sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)
    return math.sqrt(mse)


def r2_score(y_true: list[float], y_pred: list[float]) -> float:
    """Coefficient of determination (R²).

    Returns 1.0 for perfect predictions; can be negative when the model
    is worse than predicting the mean.
    """
    _check(y_true, y_pred)
    if len(y_true) < 2:
        return 0.0
    mean_true = sum(y_true) / len(y_true)
    ss_tot = sum((t - mean_true) ** 2 for t in y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1.0 - ss_res / ss_tot


def _check(a: list, b: list) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"y_true and y_pred must have the same length ({len(a)} vs {len(b)})"
        )
