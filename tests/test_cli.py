"""Smoke tests for the pocket-ml CLI."""

import os
import tempfile
import pytest

from pocket_ml_lab.cli import main
from pocket_ml_lab.experiment import ExperimentConfig, run_experiment


IRIS_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "iris_small.csv")
HOUSE_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "house_prices_small.csv")


def test_profile_iris(capsys):
    ret = main(["profile", IRIS_CSV, "--target", "species"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "species" in out
    assert "Rows" in out


def test_run_classification_iris(capsys):
    ret = main(["run", IRIS_CSV, "--target", "species", "--task", "classification"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "MajorityClassifier" in out
    assert "NearestCentroidClassifier" in out
    assert "Accuracy" in out
    assert "Balanced" in out
    assert "Best" in out


def test_run_regression_house(capsys):
    ret = main(["run", HOUSE_CSV, "--target", "price", "--task", "regression"])
    assert ret == 0
    out = capsys.readouterr().out
    assert "MeanRegressor" in out
    assert "MAE" in out


def test_run_writes_reports(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        ret = main([
            "run", IRIS_CSV,
            "--target", "species",
            "--task", "classification",
            "--out", tmpdir,
        ])
        assert ret == 0
        files = os.listdir(tmpdir)
        assert any(f.endswith(".json") for f in files)
        assert any(f.endswith(".md") for f in files)
        assert any(f.endswith(".html") for f in files)


def test_run_missing_target_column(capsys):
    ret = main(["run", IRIS_CSV, "--target", "nonexistent", "--task", "classification"])
    assert ret == 1


def test_run_missing_csv(capsys):
    ret = main(["run", "no_such_file.csv", "--target", "x", "--task", "classification"])
    assert ret == 1


def test_run_custom_seed_and_split(capsys):
    ret = main([
        "run", IRIS_CSV,
        "--target", "species",
        "--task", "classification",
        "--test-size", "0.3",
        "--seed", "7",
    ])
    assert ret == 0
    out = capsys.readouterr().out
    assert "seed=7" in out


def test_classification_experiment_ranks_models_by_balanced_accuracy():
    results = run_experiment(ExperimentConfig(IRIS_CSV, "species", "classification"))
    summary = results["evaluation"]

    assert summary["primary_metric"] == "balanced_accuracy"
    assert summary["direction"] == "higher"
    assert summary["best_model"] == "NearestCentroidClassifier"
    assert summary["rankings"][0]["score"] >= summary["rankings"][-1]["score"]


def test_regression_experiment_ranks_models_by_rmse():
    results = run_experiment(ExperimentConfig(HOUSE_CSV, "price", "regression"))
    summary = results["evaluation"]

    assert summary["primary_metric"] == "rmse"
    assert summary["direction"] == "lower"
    assert summary["best_model"] == "MeanRegressor"
    assert summary["best_score"] == results["results"][0]["metrics"]["rmse"]
