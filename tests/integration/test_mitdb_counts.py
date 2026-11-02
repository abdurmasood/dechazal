"""Integration tests against the real MIT-BIH records.

Run with: MITDB_DIR=data/raw/mitdb uv run pytest -m mitdb
"""

import os
from collections import Counter
from pathlib import Path

import pytest

from arrhythmia.config import DatasetConfig
from arrhythmia.data.dataset import build_beat_set
from arrhythmia.data.labels import AamiClass, to_aami
from arrhythmia.data.records import load_record
from arrhythmia.data.splits import DS1, DS2, TEST_RECORDS, TRAIN_RECORDS, VAL_RECORDS

pytestmark = pytest.mark.mitdb

# de Chazal et al. (2004), Table I: AAMI class counts per split (N, S, V, F, Q).
DE_CHAZAL_COUNTS = {
    "DS1": (45866, 944, 3788, 415, 8),
    "DS2": (44259, 1837, 3221, 388, 7),
}
MAX_OUT_OF_BOUNDS_PER_RECORD = 2


@pytest.fixture(scope="module")
def mitdb_dir() -> Path:
    path = os.environ.get("MITDB_DIR")
    if not path:
        pytest.skip("MITDB_DIR not set")
    return Path(path)


@pytest.mark.parametrize(("split_name", "records"), [("DS1", DS1), ("DS2", DS2)])
def test_annotation_counts_match_de_chazal(
    mitdb_dir: Path, split_name: str, records: frozenset[int]
) -> None:
    # Act
    counts: Counter[AamiClass] = Counter()
    for record_id in records:
        for symbol in load_record(mitdb_dir, record_id).symbols:
            aami = to_aami(symbol)
            if aami is not None:
                counts[aami] += 1

    # Assert
    assert tuple(counts[c] for c in AamiClass) == DE_CHAZAL_COUNTS[split_name]


def test_beat_sets_keep_patients_disjoint_and_lose_few_beats(mitdb_dir: Path) -> None:
    # Arrange
    cfg = DatasetConfig()
    groups = {"train": TRAIN_RECORDS, "val": VAL_RECORDS, "test": TEST_RECORDS}

    # Act
    beat_sets = {name: build_beat_set(mitdb_dir, ids, cfg) for name, ids in groups.items()}

    # Assert
    seen: set[int] = set()
    for name, beats in beat_sets.items():
        ids = set(beats.record_ids.tolist())
        assert ids == set(groups[name])
        assert seen.isdisjoint(ids)
        seen |= ids
        assert beats.excluded_out_of_bounds <= MAX_OUT_OF_BOUNDS_PER_RECORD * len(groups[name])
    test_beats = beat_sets["test"]
    n, s, v, f, q = DE_CHAZAL_COUNTS["DS2"]
    assert len(test_beats) + test_beats.excluded_out_of_bounds == n + s + v + f
    assert test_beats.excluded_q == q
