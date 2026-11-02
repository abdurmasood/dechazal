"""Tests for per-class metrics and multi-seed aggregation."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from arrhythmia.training.evaluation import (
    aggregate_runs,
    evaluate_predictions,
    run_metrics_from_dict,
    run_metrics_to_dict,
    to_markdown,
)

CLASS_NAMES = ("N", "S", "V", "F")
Y_TRUE = np.array([0, 0, 0, 1, 1, 2, 3])
Y_PRED = np.array([0, 0, 1, 1, 0, 2, 2])


def test_per_class_metrics_match_hand_computed_confusion() -> None:
    # Act
    run = evaluate_predictions(Y_TRUE, Y_PRED, CLASS_NAMES)

    # Assert: class N has TP=2, FN=1, FP=1, TN=3
    n = run.per_class[0]
    assert n.support == 3
    assert n.se == pytest.approx(2 / 3)
    assert n.ppv == pytest.approx(2 / 3)
    assert n.fpr == pytest.approx(1 / 4)
    assert run.confusion[0] == [2, 1, 0, 0]


def test_macro_f1_matches_sklearn() -> None:
    run = evaluate_predictions(Y_TRUE, Y_PRED, CLASS_NAMES)

    expected = f1_score(Y_TRUE, Y_PRED, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    assert run.macro_f1 == pytest.approx(expected)


def test_class_never_predicted_reports_ppv_zero_and_flags_it() -> None:
    run = evaluate_predictions(Y_TRUE, Y_PRED, CLASS_NAMES)

    f = run.per_class[3]
    assert f.never_predicted
    assert f.ppv == 0.0
    assert f.f1 == 0.0
    assert not run.per_class[0].never_predicted


def test_run_metrics_round_trip_through_dict() -> None:
    run = evaluate_predictions(Y_TRUE, Y_PRED, CLASS_NAMES)

    assert run_metrics_from_dict(run_metrics_to_dict(run)) == run


def test_aggregate_runs_reports_mean_and_std() -> None:
    # Arrange: two seeds with different predictions
    run_a = evaluate_predictions(Y_TRUE, Y_PRED, CLASS_NAMES)
    run_b = evaluate_predictions(Y_TRUE, Y_TRUE, CLASS_NAMES)

    # Act
    summary = aggregate_runs([run_a, run_b])

    # Assert: sample std (ddof=1) across seeds
    mean, std = summary["macro_f1"]
    assert mean == pytest.approx((run_a.macro_f1 + 1.0) / 2)
    assert std == pytest.approx(np.std([run_a.macro_f1, 1.0], ddof=1))
    assert summary["N.se"][0] == pytest.approx((2 / 3 + 1.0) / 2)


def test_aggregate_single_run_has_zero_std() -> None:
    summary = aggregate_runs([evaluate_predictions(Y_TRUE, Y_PRED, CLASS_NAMES)])

    assert summary["macro_f1"][1] == 0.0


def test_to_markdown_lists_each_model_with_mean_and_std() -> None:
    run = evaluate_predictions(Y_TRUE, Y_PRED, CLASS_NAMES)

    table = to_markdown({"cnn1d": ([run], aggregate_runs([run]))}, CLASS_NAMES)

    assert "| cnn1d |" in table
    assert "±" in table
    assert "S Se" in table
