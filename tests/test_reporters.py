"""Regression tests for the JSON/Markdown/HTML reporters.

Focus on the HTML report's screen-reader accessibility: every table must
announce what it contains (`<caption>`) and how its cells relate to each
other (`scope="col"` / `scope="row"`), per WCAG technique H43/H63.
"""

import json
import os
import re
import tempfile

from pocket_ml_lab.experiment import ExperimentConfig, run_experiment
from pocket_ml_lab.reporters.json_reporter import format_json, write_json_report
from pocket_ml_lab.reporters.markdown_reporter import write_html_report, write_markdown_report


IRIS_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "iris_small.csv")
HOUSE_CSV = os.path.join(os.path.dirname(__file__), "..", "examples", "house_prices_small.csv")


def _classification_results():
    return run_experiment(ExperimentConfig(IRIS_CSV, "species", "classification"))


def _regression_results():
    return run_experiment(ExperimentConfig(HOUSE_CSV, "price", "regression"))


def test_html_report_tables_have_captions():
    html = _render_html(_classification_results())
    table_count = html.count("<table>")
    caption_count = len(re.findall(r"<caption>", html))
    assert table_count > 0
    # Every table in a classification report is preceded by a heading or
    # bold label, so every table should get a caption.
    assert caption_count == table_count


def test_html_report_header_rows_use_scope_col():
    html = _render_html(_classification_results())
    assert '<th scope="col">Key</th>' in html
    assert '<th scope="col">Value</th>' in html


def test_html_report_data_rows_use_scope_row_for_first_cell():
    html = _render_html(_classification_results())
    assert '<th scope="row">Dataset</th>' in html
    assert '<th scope="row">Target</th>' in html
    # Confusion-matrix / per-class rows are keyed by class name, not "Key".
    assert '<th scope="row">setosa</th>' in html or '<th scope="row">versicolor</th>' in html


def test_html_report_regression_table_also_captioned():
    html = _render_html(_regression_results())
    assert "<caption>MeanRegressor</caption>" in html
    assert '<th scope="row">MAE</th>' in html


def test_markdown_report_still_renders_plain_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_markdown_report(_classification_results(), os.path.join(tmpdir, "r.md"))
        text = open(path, encoding="utf-8").read()
        assert "| Dataset |" in text
        assert "<caption>" not in text  # Markdown output must not leak HTML


def test_json_report_round_trips():
    results = _regression_results()
    text = format_json(results)
    parsed = json.loads(text)
    assert parsed["evaluation"]["primary_metric"] == "rmse"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_json_report(results, os.path.join(tmpdir, "r.json"))
        assert json.load(open(path, encoding="utf-8"))["dataset"]["n_rows"] > 0


def _render_html(results) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = write_html_report(results, os.path.join(tmpdir, "r.html"))
        return open(path, encoding="utf-8").read()
