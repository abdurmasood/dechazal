"""Tests for building labelled beat sets from records."""

from pathlib import Path

import numpy as np
import pytest

from arrhythmia.config import DatasetConfig
from arrhythmia.data import dataset
from arrhythmia.data.dataset import (
    build_beat_set,
    fit_standardizer,
    load_or_build_beat_set,
    standardize,
)
from arrhythmia.data.labels import AamiClass


def test_beat_set_carries_record_id_per_beat(write_record, tmp_path: Path) -> None:
    # Arrange
    write_record(100, ["N", "N", "V"])
    write_record(101, ["N", "A"])

    # Act
    beats = build_beat_set(tmp_path, frozenset({100, 101}), DatasetConfig())

    # Assert
    assert beats.record_ids.tolist() == [100, 100, 100, 101, 101]
    assert beats.signals.shape == (5, 252)
    assert beats.rr.shape == (5, 4)


def test_labels_follow_aami_mapping_and_skip_non_beats(write_record, tmp_path: Path) -> None:
    # Arrange: the fixture also writes a leading '+' rhythm annotation
    write_record(100, ["N", "A", "V", "F"])

    # Act
    beats = build_beat_set(tmp_path, frozenset({100}), DatasetConfig())

    # Assert
    assert beats.labels.tolist() == [AamiClass.N, AamiClass.S, AamiClass.V, AamiClass.F]


def test_q_beats_excluded_and_counted(write_record, tmp_path: Path) -> None:
    write_record(100, ["N", "Q", "N"])

    beats = build_beat_set(tmp_path, frozenset({100}), DatasetConfig())

    assert beats.excluded_q == 1
    assert AamiClass.Q not in beats.labels.tolist()
    assert len(beats.labels) == 2


def test_cache_hit_returns_identical_arrays(
    write_record, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    write_record(100, ["N", "V", "N"])
    cache_dir = tmp_path / "cache"
    first = load_or_build_beat_set(tmp_path, frozenset({100}), DatasetConfig(), cache_dir)

    def _fail(*_: object) -> None:
        raise AssertionError("cache miss: record was re-read")

    monkeypatch.setattr(dataset, "load_record", _fail)

    # Act
    second = load_or_build_beat_set(tmp_path, frozenset({100}), DatasetConfig(), cache_dir)

    # Assert
    np.testing.assert_array_equal(first.signals, second.signals)
    np.testing.assert_array_equal(first.labels, second.labels)
    np.testing.assert_array_equal(first.rr, second.rr)
    assert first.excluded_q == second.excluded_q


def test_standardizer_uses_only_training_statistics(write_record, tmp_path: Path) -> None:
    # Arrange
    write_record(100, ["N", "V", "N", "V"])
    train = build_beat_set(tmp_path, frozenset({100}), DatasetConfig())

    # Act
    mean, std = fit_standardizer(train)
    scaled = standardize(train, mean, std)

    # Assert
    assert scaled.signals.mean() == pytest.approx(0.0, abs=1e-5)
    assert scaled.signals.std() == pytest.approx(1.0, rel=1e-4)
    np.testing.assert_array_equal(scaled.labels, train.labels)
