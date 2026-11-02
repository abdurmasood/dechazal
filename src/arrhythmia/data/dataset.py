"""Turning MIT-BIH records into labelled, patient-tagged beat arrays."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from arrhythmia.config import DatasetConfig
from arrhythmia.data.labels import AamiClass, to_aami
from arrhythmia.data.preprocessing import extract_windows, filter_signal
from arrhythmia.data.records import Record, load_record
from arrhythmia.data.rr import RR_FEATURE_NAMES, rr_features

_ARRAY_FIELDS = ("signals", "labels", "record_ids", "r_samples", "rr")


@dataclass(frozen=True)
class BeatSet:
    """Beats from one or more records, every row tagged with its patient.

    Attributes:
        signals: Filtered beat windows, shape (n, window_len), float32.
        labels: AAMI class index (N=0, S=1, V=2, F=3), shape (n,), int8.
        record_ids: Source record of each beat, shape (n,), int16.
        r_samples: R-peak sample index within its record, shape (n,), int64.
        rr: RR features in RR_FEATURE_NAMES order, shape (n, 4), float32.
        excluded_q: Q-class beats dropped (too rare to learn).
        excluded_out_of_bounds: Beats dropped because the window ran off the record.
    """

    signals: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    r_samples: np.ndarray
    rr: np.ndarray
    excluded_q: int = 0
    excluded_out_of_bounds: int = 0

    def __len__(self) -> int:
        return int(self.labels.size)


def build_record_beats(record: Record, cfg: DatasetConfig) -> BeatSet:
    """Filter, window and label all beats of one record.

    Args:
        record: Loaded record.
        cfg: Filter and window settings.

    Returns:
        The record's N/S/V/F beats with exclusion counts.
    """
    classes = [to_aami(symbol) for symbol in record.symbols]
    is_beat = np.array([c is not None for c in classes])
    beat_samples = record.ann_samples[is_beat]
    beat_classes = np.array([int(c) for c in classes if c is not None], dtype=np.int8)
    # RR uses every beat, Q included, so dropping Q later does not stretch intervals.
    rr = rr_features(beat_samples, record.fs)
    filtered = filter_signal(record.signal, record.fs, cfg.filter)
    windows, in_bounds = extract_windows(filtered, beat_samples, record.fs, cfg.window)
    not_q = beat_classes[in_bounds] != AamiClass.Q
    kept = np.flatnonzero(in_bounds)[not_q]
    return BeatSet(
        signals=windows[not_q].astype(np.float32),
        labels=beat_classes[kept],
        record_ids=np.full(kept.size, record.record_id, dtype=np.int16),
        r_samples=beat_samples[kept],
        rr=rr[kept],
        excluded_q=int(np.sum(beat_classes == AamiClass.Q)),
        excluded_out_of_bounds=int(np.sum(~in_bounds & (beat_classes != AamiClass.Q))),
    )


def concat_beat_sets(parts: list[BeatSet]) -> BeatSet:
    """Concatenate beat sets row-wise and sum their exclusion counts."""
    arrays = {name: np.concatenate([getattr(p, name) for p in parts]) for name in _ARRAY_FIELDS}
    return BeatSet(
        **arrays,
        excluded_q=sum(p.excluded_q for p in parts),
        excluded_out_of_bounds=sum(p.excluded_out_of_bounds for p in parts),
    )


def build_beat_set(data_dir: Path, record_ids: frozenset[int], cfg: DatasetConfig) -> BeatSet:
    """Build the beat set for a group of records, in ascending record order."""
    parts = [build_record_beats(load_record(data_dir, rid), cfg) for rid in sorted(record_ids)]
    return concat_beat_sets(parts)


def load_or_build_beat_set(
    data_dir: Path, record_ids: frozenset[int], cfg: DatasetConfig, cache_dir: Path
) -> BeatSet:
    """Return a cached beat set, building and caching it on first use.

    The cache key hashes the config and the record IDs, so changing either rebuilds.
    """
    cache_path = Path(cache_dir) / f"beats_{cfg.digest(record_ids)}.npz"
    if cache_path.exists():
        with np.load(cache_path) as cached:
            return BeatSet(
                **{name: cached[name] for name in _ARRAY_FIELDS},
                excluded_q=int(cached["excluded_q"]),
                excluded_out_of_bounds=int(cached["excluded_out_of_bounds"]),
            )
    beats = build_beat_set(data_dir, record_ids, cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        **{name: getattr(beats, name) for name in _ARRAY_FIELDS},
        excluded_q=beats.excluded_q,
        excluded_out_of_bounds=beats.excluded_out_of_bounds,
    )
    return beats


def fit_standardizer(train: BeatSet) -> tuple[float, float]:
    """Global scalar mean and std of the training signals (fitted on training patients only).

    A single scalar keeps relative amplitude between beats, which carries V-beat information.
    """
    return float(train.signals.mean()), float(train.signals.std())


def standardize(beats: BeatSet, mean: float, std: float) -> BeatSet:
    """Return a copy of beats with signals scaled by training statistics."""
    return replace(beats, signals=((beats.signals - mean) / std).astype(np.float32))


__all__ = [
    "RR_FEATURE_NAMES",
    "BeatSet",
    "build_beat_set",
    "build_record_beats",
    "concat_beat_sets",
    "fit_standardizer",
    "load_or_build_beat_set",
    "standardize",
]
