"""End-to-end steps shared by the CLI: build splits, train one seed, collect results."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from arrhythmia.config import DatasetConfig, ImageConfig
from arrhythmia.data.dataset import BeatSet, fit_standardizer, load_or_build_beat_set, standardize
from arrhythmia.data.images import load_or_render_images
from arrhythmia.data.labels import TRAINING_CLASSES
from arrhythmia.data.splits import TEST_RECORDS, TRAIN_RECORDS, VAL_RECORDS
from arrhythmia.models import ModelKind, build_model, get_model_spec
from arrhythmia.training.evaluation import (
    RunMetrics,
    evaluate_predictions,
    run_metrics_from_dict,
    run_metrics_to_dict,
)
from arrhythmia.training.fit import (
    SplitArrays,
    SplitData,
    TrainConfig,
    fit_model,
    predict_classes,
    set_seed,
)

SPLIT_RECORDS: dict[str, frozenset[int]] = {
    "train": TRAIN_RECORDS,
    "val": VAL_RECORDS,
    "test": TEST_RECORDS,
}
CLASS_NAMES: tuple[str, ...] = tuple(c.name for c in TRAINING_CLASSES)


def load_splits(data_dir: Path, cache_dir: Path, cfg: DatasetConfig) -> dict[str, BeatSet]:
    """Build (or load cached) train/val/test beat sets, standardised with train statistics."""
    raw = {
        name: load_or_build_beat_set(data_dir, records, cfg, cache_dir)
        for name, records in SPLIT_RECORDS.items()
    }
    mean, std = fit_standardizer(raw["train"])
    return {name: standardize(beats, mean, std) for name, beats in raw.items()}


def _main_input(beats: BeatSet, kind: ModelKind, image_path: Path) -> np.ndarray:
    if kind is ModelKind.IMAGE_2D:
        return load_or_render_images(beats.signals, image_path, ImageConfig())
    return beats.signals


def to_split_data(
    splits: dict[str, BeatSet], kind: ModelKind, image_dir: Path, cfg: DatasetConfig
) -> SplitData:
    """Arrange beat sets as model inputs for the given model kind."""
    arrays = {}
    for name, beats in splits.items():
        image_path = image_dir / f"images_{name}_{cfg.digest(SPLIT_RECORDS[name])}.u8"
        main = _main_input(beats, kind, image_path)
        inputs = (main, beats.rr) if kind is ModelKind.SIGNAL_1D_RR else (main,)
        arrays[name] = SplitArrays(inputs, beats.labels, beats.record_ids)
    return SplitData(**arrays)


def _write_run(run_dir: Path, run: RunMetrics, run_info: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(run_metrics_to_dict(run), indent=2))
    (run_dir / "config.json").write_text(json.dumps(run_info, indent=2))
    header = "true\\pred," + ",".join(CLASS_NAMES)
    rows = [
        f"{name}," + ",".join(map(str, row))
        for name, row in zip(CLASS_NAMES, run.confusion, strict=True)
    ]
    (run_dir / "confusion.csv").write_text("\n".join([header, *rows]) + "\n")


def train_one_seed(model_name: str, data: SplitData, cfg: TrainConfig, run_dir: Path) -> RunMetrics:
    """Train one seed, evaluate on the test patients and write its artifacts to run_dir."""
    spec = get_model_spec(model_name)
    set_seed(cfg.seed)
    model = build_model(model_name, len(CLASS_NAMES))
    history = fit_model(model, data, cfg, spec.batch_size)
    predicted = predict_classes(model, data.test, spec.batch_size)
    run = evaluate_predictions(data.test.labels, predicted, CLASS_NAMES)
    run_info = {
        "model": model_name,
        "train_config": asdict(cfg),
        "epochs_run": len(history.history["loss"]),
        "records": {name: sorted(ids) for name, ids in SPLIT_RECORDS.items()},
        "n_beats": {name: int(getattr(data, name).labels.size) for name in SPLIT_RECORDS},
    }
    _write_run(run_dir, run, run_info)
    model.save(run_dir / "model.keras")
    return run


def dataset_summary(splits: dict[str, BeatSet]) -> dict[str, dict[str, int]]:
    """Per-split beat counts per class plus exclusions, for logs and the README."""
    summary = {}
    for name, beats in splits.items():
        counts = np.bincount(beats.labels, minlength=len(CLASS_NAMES))
        summary[name] = {
            **{cls: int(n) for cls, n in zip(CLASS_NAMES, counts, strict=True)},
            "excluded_q": beats.excluded_q,
            "excluded_out_of_bounds": beats.excluded_out_of_bounds,
        }
    return summary


def collect_runs(runs_dir: Path) -> dict[str, list[RunMetrics]]:
    """Load metrics.json from runs_dir/<model>/seed*/, or runs_dir/seed*/ for one model."""
    runs_dir = Path(runs_dir)
    model_dirs = [runs_dir] if any(runs_dir.glob("seed*")) else sorted(runs_dir.iterdir())
    results: dict[str, list[RunMetrics]] = {}
    for model_dir in model_dirs:
        metric_files = sorted(model_dir.glob("seed*/metrics.json"))
        if metric_files:
            results[model_dir.name] = [
                run_metrics_from_dict(json.loads(f.read_text())) for f in metric_files
            ]
    return results
