"""Command line entry point: arrhythmia download | prepare | train | report."""

import argparse
import json
from pathlib import Path

from arrhythmia.config import DatasetConfig
from arrhythmia.data.records import download_mitdb
from arrhythmia.models import MODEL_REGISTRY, get_model_spec
from arrhythmia.pipeline import (
    CLASS_NAMES,
    collect_runs,
    dataset_summary,
    load_splits,
    to_split_data,
    train_one_seed,
)
from arrhythmia.training.evaluation import aggregate_runs, to_markdown
from arrhythmia.training.fit import TrainConfig

DEFAULT_DATA_DIR = Path("data/raw/mitdb")
DEFAULT_CACHE_DIR = Path("data/processed")
DEFAULT_RUNS_DIR = Path("runs")
DATASET_SUMMARY_FILE = "dataset.json"


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arrhythmia", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="fetch MIT-BIH from PhysioNet")
    download.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)

    prepare = commands.add_parser("prepare", help="filter, window and cache all splits")
    _add_data_args(prepare)

    train = commands.add_parser("train", help="train one model for one or more seeds")
    _add_data_args(train)
    train.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    train.add_argument("--seeds", type=int, nargs="+", default=[0])
    train.add_argument("--epochs", type=int, default=TrainConfig().epochs)
    train.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    report = commands.add_parser("report", help="summarise runs as a markdown table")
    report.add_argument("runs_dir", type=Path, nargs="?", default=DEFAULT_RUNS_DIR)
    return parser


def _prepare(args: argparse.Namespace) -> int:
    splits = load_splits(args.data_dir, args.cache_dir, DatasetConfig())
    print(json.dumps(dataset_summary(splits), indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    cfg = DatasetConfig()
    splits = load_splits(args.data_dir, args.cache_dir, cfg)
    args.runs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.runs_dir / DATASET_SUMMARY_FILE
    summary_path.write_text(json.dumps(dataset_summary(splits), indent=2))
    kind = get_model_spec(args.model).kind
    data = to_split_data(splits, kind, args.cache_dir, cfg)
    for seed in args.seeds:
        run_dir = args.runs_dir / args.model / f"seed{seed}"
        run = train_one_seed(args.model, data, TrainConfig(epochs=args.epochs, seed=seed), run_dir)
        print(f"{args.model} seed {seed}: macro-F1 {run.macro_f1:.3f}")
    return 0


def _find_dataset_summary(runs_dir: Path) -> dict | None:
    for candidate in (runs_dir / DATASET_SUMMARY_FILE, runs_dir.parent / DATASET_SUMMARY_FILE):
        if candidate.exists():
            return json.loads(candidate.read_text())
    return None


def _report(args: argparse.Namespace) -> int:
    runs = collect_runs(args.runs_dir)
    if not runs:
        print(f"No runs found under {args.runs_dir}")
        return 1
    results = {name: (model_runs, aggregate_runs(model_runs)) for name, model_runs in runs.items()}
    print(to_markdown(results, CLASS_NAMES))
    summary = _find_dataset_summary(args.runs_dir)
    if summary is not None:
        ds1_q = summary["train"]["excluded_q"] + summary["val"]["excluded_q"]
        print(f"\nQ excluded: {ds1_q} (DS1), {summary['test']['excluded_q']} (DS2)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    args = _build_parser().parse_args(argv)
    if args.command == "download":
        download_mitdb(args.data_dir)
        return 0
    handlers = {"prepare": _prepare, "train": _train, "report": _report}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
