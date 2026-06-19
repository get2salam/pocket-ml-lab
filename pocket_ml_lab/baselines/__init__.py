"""Baseline models: majority classifier, mean regressor, nearest-centroid classifier."""

from .majority_classifier import MajorityClassifier
from .mean_regressor import MeanRegressor
from .nearest_centroid import NearestCentroidClassifier

__all__ = ["MajorityClassifier", "MeanRegressor", "NearestCentroidClassifier"]
