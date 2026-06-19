"""pocket-ml CLI entry point."""

import argparse
import os
import sys

from .experiment import ExperimentConfig, run_experiment
from .loader import load_csv
from .profiler import profile_dataset
from .reporters.json_reporter import write_json_report, format_json
from .reporters.markdown_reporter import write_markdown_report, write_html_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pocket-ml",
        description="Pocket ML Lab — lightweight baseline ML experiments on CSV datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── profile ──────────────────────────────────────────────────────────────
    p_profile = subparsers.add_parser("profile", help="Show dataset schema and statistics")
    p_profile.add_argument("csv", help="Path to CSV file")
    p_profile.add_argument("--target", help="Highlight this column as the target")

    # ── run ──────────────────────────────────────────────────────────────────
    p_run = subparsers.add_parser("run", help="Run baseline experiments")
    p_run.add_argument("csv", help="Path to CSV file")
    p_run.add_argument("--target", required=True, help="Target column name")
    p_run.add_argument(
        "--task",
        choices=["classification", "regression"],
        default="classification",
        help="Task type (default: classification)",
    )
    p_run.add_argument("--test-size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    p_run.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_run.add_argument(
        "--out",
        metavar="DIR",
        help="Write JSON + Markdown + HTML reports to this directory",
    )

    args = parser.parse_args(argv)

    if args.command == "profile":
        return _cmd_profile(args)
    if args.command == "run":
        return _cmd_run(args)

    parser.print_help()
    return 1


def _cmd_profile(args: argparse.Namespace) -> int:
    try:
        rows = load_csv(args.csv)
    except (FileNotFoundError, ValueError) as exc:
        _err(str(exc))
        return 1

    profile = profile_dataset(rows)
    print(f"\n=== Dataset Profile: {args.csv} ===\n")
    print(f"  Rows    : {profile['n_rows']}")
    print(f"  Columns : {profile['n_cols']}")
    print()

    for col in profile["columns"]:
        p = profile["profiles"][col]
        tag = "[TARGET]" if args.target and col == args.target else ""
        print(f"  {col} {tag}")
        print(f"    type      : {p['dtype']}")
        print(f"    nulls     : {p['null_count']} ({p['null_pct']}%)")
        if p["dtype"] == "numeric":
            print(f"    min/max   : {p['min']} / {p['max']}")
            print(f"    mean ± std: {p['mean']} ± {p['std']}")
            print(f"    median    : {p['median']}")
        else:
            print(f"    cardinality: {p.get('cardinality', '?')}")
            print(f"    top values : {p.get('top', [])}")
        print()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    config = ExperimentConfig(
        csv_path=args.csv,
        target=args.target,
        task=args.task,
        test_size=args.test_size,
        seed=args.seed,
    )

    try:
        results = run_experiment(config)
    except (FileNotFoundError, ValueError) as exc:
        _err(str(exc))
        return 1

    ds = results["dataset"]
    print(f"\n=== Pocket ML Lab ===")
    print(f"Dataset : {args.csv}")
    print(f"Task    : {args.task}")
    print(f"Target  : {args.target}")
    print(f"Split   : {ds['n_train']} train / {ds['n_test']} test (seed={args.seed})")
    print()

    for mr in results["results"]:
        print(f"--- {mr['model']} ---")
        metrics = mr["metrics"]
        if args.task == "classification":
            print(f"  Accuracy  : {metrics['accuracy']}")
            print(f"  Macro F1  : {metrics['macro_f1']}")
            print(f"  Macro P   : {metrics['macro_precision']}")
            print(f"  Macro R   : {metrics['macro_recall']}")
        else:
            print(f"  MAE       : {metrics['mae']}")
            print(f"  RMSE      : {metrics['rmse']}")
            print(f"  R²        : {metrics['r2']}")
        print()

    if args.out:
        try:
            _write_reports(results, args.out, args.target, args.task)
        except OSError as exc:
            _err(f"Could not write reports: {exc}")
            return 1

    return 0


def _write_reports(
    results: dict, out_dir: str, target: str, task: str
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"experiment_{target}_{task}"
    json_path = os.path.join(out_dir, f"{stem}.json")
    md_path = os.path.join(out_dir, f"{stem}.md")
    html_path = os.path.join(out_dir, f"{stem}.html")

    write_json_report(results, json_path)
    write_markdown_report(results, md_path)
    write_html_report(results, html_path)

    print(f"Reports written to {out_dir}/")
    print(f"  {os.path.basename(json_path)}")
    print(f"  {os.path.basename(md_path)}")
    print(f"  {os.path.basename(html_path)}")


def _err(msg: str) -> None:
    print(f"[pocket-ml] ERROR: {msg}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
