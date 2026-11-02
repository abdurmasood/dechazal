"""Per-class beat classification metrics, as reported in the inter-patient literature.

Plain Se/PPV/FPR from the raw confusion matrix. ANSI/AAMI EC57 additionally exempts F beats
classified as V from V's false positives; that convention is not applied here.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

MeanStd = tuple[float, float]


@dataclass(frozen=True)
class ClassMetrics:
    """One-vs-rest metrics for a single class."""

    name: str
    support: int
    se: float  # sensitivity (recall)
    ppv: float  # positive predictive value (precision)
    fpr: float  # false positive rate
    f1: float
    never_predicted: bool


@dataclass(frozen=True)
class RunMetrics:
    """Metrics of one trained model (one seed) on the test patients."""

    per_class: tuple[ClassMetrics, ...]
    macro_f1: float
    accuracy: float
    confusion: list[list[int]]  # rows = true class, columns = predicted class


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _class_metrics(cm: np.ndarray, index: int, name: str) -> ClassMetrics:
    tp = float(cm[index, index])
    fn = float(cm[index].sum() - tp)
    fp = float(cm[:, index].sum() - tp)
    tn = float(cm.sum() - tp - fn - fp)
    se, ppv = _safe_div(tp, tp + fn), _safe_div(tp, tp + fp)
    return ClassMetrics(
        name=name,
        support=int(tp + fn),
        se=se,
        ppv=ppv,
        fpr=_safe_div(fp, fp + tn),
        f1=_safe_div(2 * se * ppv, se + ppv),
        never_predicted=bool(tp + fp == 0),
    )


def evaluate_predictions(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: tuple[str, ...]
) -> RunMetrics:
    """Compute per-class and macro metrics from integer labels.

    Args:
        y_true: True class indices.
        y_pred: Predicted class indices.
        class_names: Name of each class index, in order.

    Returns:
        Metrics including the full confusion matrix.
    """
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    per_class = tuple(_class_metrics(cm, i, name) for i, name in enumerate(class_names))
    return RunMetrics(
        per_class=per_class,
        macro_f1=float(np.mean([c.f1 for c in per_class])),
        accuracy=_safe_div(float(np.trace(cm)), float(cm.sum())),
        confusion=cm.tolist(),
    )


def run_metrics_to_dict(run: RunMetrics) -> dict[str, Any]:
    """Serialise for metrics.json."""
    return asdict(run)


def run_metrics_from_dict(data: dict[str, Any]) -> RunMetrics:
    """Inverse of run_metrics_to_dict."""
    return RunMetrics(
        per_class=tuple(ClassMetrics(**c) for c in data["per_class"]),
        macro_f1=float(data["macro_f1"]),
        accuracy=float(data["accuracy"]),
        confusion=[list(map(int, row)) for row in data["confusion"]],
    )


def aggregate_runs(runs: list[RunMetrics]) -> dict[str, MeanStd]:
    """Mean and sample std (ddof=1) of every metric across seeds.

    Keys are "macro_f1", "accuracy" and "<class>.<metric>" (e.g. "S.se"). Seed variance is
    not patient variance: it says nothing about how results move with other test patients.
    """
    values: dict[str, list[float]] = {"macro_f1": [], "accuracy": []}
    for run in runs:
        values["macro_f1"].append(run.macro_f1)
        values["accuracy"].append(run.accuracy)
        for c in run.per_class:
            for metric in ("se", "ppv", "fpr", "f1"):
                values.setdefault(f"{c.name}.{metric}", []).append(getattr(c, metric))
    ddof = 1 if len(runs) > 1 else 0
    return {k: (float(np.mean(v)), float(np.std(v, ddof=ddof))) for k, v in values.items()}


def _cell(summary: dict[str, MeanStd], key: str) -> str:
    mean, std = summary[key]
    return f"{100 * mean:.1f} ± {100 * std:.1f}"


def to_markdown(
    results: dict[str, tuple[list[RunMetrics], dict[str, MeanStd]]],
    class_names: tuple[str, ...],
) -> str:
    """Render a results table: one row per model, values in percent as mean ± std.

    Args:
        results: Model name mapped to (its runs, aggregate_runs of those runs).
        class_names: Classes to report Se and PPV for.

    Returns:
        A GitHub-flavoured markdown table.
    """
    columns = ["Macro-F1"] + [f"{c} {m}" for c in class_names for m in ("Se", "PPV")]
    header = "| Model | Seeds | " + " | ".join(columns) + " |"
    divider = "|" + "---|" * (len(columns) + 2)
    rows = [header, divider]
    for name, (runs, summary) in results.items():
        keys = ["macro_f1"] + [f"{c}.{m}" for c in class_names for m in ("se", "ppv")]
        cells = " | ".join(_cell(summary, k) for k in keys)
        rows.append(f"| {name} | {len(runs)} | {cells} |")
    return "\n".join(rows)
