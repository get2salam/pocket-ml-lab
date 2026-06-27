"""Tests for baseline models: majority classifier, mean/median regressors, nearest centroid."""

import pytest

from pocket_ml_lab.baselines.majority_classifier import MajorityClassifier
from pocket_ml_lab.baselines.mean_regressor import MeanRegressor
from pocket_ml_lab.baselines.median_regressor import MedianRegressor
from pocket_ml_lab.baselines.nearest_centroid import NearestCentroidClassifier


# ── MajorityClassifier ────────────────────────────────────────────────────────

def _rows(n):
    return [{}] * n


def test_majority_classifier_predicts_majority():
    clf = MajorityClassifier()
    clf.fit(_rows(5), ["a", "a", "a", "b", "b"])
    preds = clf.predict(_rows(3))
    assert all(p == "a" for p in preds)


def test_majority_classifier_all_same_class():
    clf = MajorityClassifier()
    clf.fit(_rows(3), ["x", "x", "x"])
    assert clf.predict(_rows(2)) == ["x", "x"]


def test_majority_classifier_returns_same_length():
    clf = MajorityClassifier()
    clf.fit(_rows(4), ["a", "b", "a", "b"])
    assert len(clf.predict(_rows(7))) == 7


def test_majority_classifier_predict_before_fit():
    with pytest.raises(RuntimeError):
        MajorityClassifier().predict(_rows(1))


def test_majority_classifier_empty_labels():
    with pytest.raises((ValueError, ZeroDivisionError)):
        MajorityClassifier().fit([], [])


def test_majority_classifier_describe():
    clf = MajorityClassifier()
    clf.fit(_rows(3), ["a", "a", "b"])
    d = clf.describe()
    assert d["majority_class"] == "a"
    assert "class_counts" in d


# ── MeanRegressor ─────────────────────────────────────────────────────────────

def test_mean_regressor_predicts_mean():
    reg = MeanRegressor()
    reg.fit(_rows(3), [1.0, 2.0, 3.0])
    preds = reg.predict(_rows(4))
    assert all(p == pytest.approx(2.0) for p in preds)


def test_mean_regressor_single_value():
    reg = MeanRegressor()
    reg.fit(_rows(1), [7.5])
    assert reg.predict(_rows(1)) == [pytest.approx(7.5)]


def test_mean_regressor_returns_same_length():
    reg = MeanRegressor()
    reg.fit(_rows(2), [0.0, 4.0])
    assert len(reg.predict(_rows(5))) == 5


def test_mean_regressor_predict_before_fit():
    with pytest.raises(RuntimeError):
        MeanRegressor().predict(_rows(1))


def test_mean_regressor_describe():
    reg = MeanRegressor()
    reg.fit(_rows(2), [1.0, 3.0])
    d = reg.describe()
    assert d["training_mean"] == pytest.approx(2.0)


# ── MedianRegressor ───────────────────────────────────────────────────────────

def test_median_regressor_odd_count():
    reg = MedianRegressor()
    reg.fit(_rows(5), [1.0, 3.0, 5.0, 7.0, 9.0])
    preds = reg.predict(_rows(3))
    assert all(p == pytest.approx(5.0) for p in preds)


def test_median_regressor_even_count():
    reg = MedianRegressor()
    reg.fit(_rows(4), [1.0, 2.0, 3.0, 4.0])
    preds = reg.predict(_rows(2))
    assert all(p == pytest.approx(2.5) for p in preds)


def test_median_regressor_robust_to_outlier():
    # Mean would be (1+2+3+100)/4 = 26.5; median should be 2.5
    reg = MedianRegressor()
    reg.fit(_rows(4), [1.0, 2.0, 3.0, 100.0])
    preds = reg.predict(_rows(1))
    assert preds[0] == pytest.approx(2.5)


def test_median_regressor_single_value():
    reg = MedianRegressor()
    reg.fit(_rows(1), [42.0])
    assert reg.predict(_rows(1)) == [pytest.approx(42.0)]


def test_median_regressor_returns_correct_length():
    reg = MedianRegressor()
    reg.fit(_rows(3), [1.0, 2.0, 3.0])
    assert len(reg.predict(_rows(7))) == 7


def test_median_regressor_predict_before_fit():
    with pytest.raises(RuntimeError):
        MedianRegressor().predict(_rows(1))


def test_median_regressor_no_numeric_values():
    with pytest.raises(ValueError):
        MedianRegressor().fit(_rows(2), [None, None])


def test_median_regressor_describe():
    reg = MedianRegressor()
    reg.fit(_rows(3), [1.0, 2.0, 3.0])
    d = reg.describe()
    assert d["training_median"] == pytest.approx(2.0)
    assert "model" in d


# ── NearestCentroidClassifier ─────────────────────────────────────────────────

def _make_X(points):
    return [{"x": p[0], "y": p[1]} for p in points]


def test_nearest_centroid_linearly_separable():
    X_train = _make_X([(1, 1), (2, 1), (8, 8), (9, 8)])
    y_train = ["a", "a", "b", "b"]
    clf = NearestCentroidClassifier()
    clf.fit(X_train, y_train)
    X_test = _make_X([(1.5, 1), (8.5, 8)])
    preds = clf.predict(X_test)
    assert preds == ["a", "b"]


def test_nearest_centroid_predict_before_fit():
    with pytest.raises(RuntimeError):
        NearestCentroidClassifier().predict(_make_X([(0, 0)]))


def test_nearest_centroid_no_numeric_features():
    X = [{"label": "x"}, {"label": "y"}]
    with pytest.raises(ValueError):
        NearestCentroidClassifier().fit(X, ["a", "b"])


def test_nearest_centroid_describe():
    clf = NearestCentroidClassifier()
    clf.fit(_make_X([(0, 0), (1, 1)]), ["neg", "pos"])
    d = clf.describe()
    assert "centroids" in d
    assert "features" in d
    assert set(d["features"]) == {"x", "y"}
