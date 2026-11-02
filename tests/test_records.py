"""Tests for loading MIT-BIH records."""

from pathlib import Path

import numpy as np
import pytest
import wfdb

from arrhythmia.data import records
from arrhythmia.data.records import LeadNotFoundError, download_mitdb, load_record


def test_load_record_selects_mlii_when_not_first_channel(write_record) -> None:
    # Arrange: MLII is channel 1, as in MIT-BIH record 114
    data_dir = write_record(114, ["N", "V", "N"], sig_names=("V5", "MLII"))

    # Act
    record = load_record(data_dir, 114)

    # Assert: the constant 0.5 mV V5 channel was not picked
    assert record.signal.std() > 0.1
    assert record.fs == 360
    assert record.record_id == 114


def test_load_record_returns_all_annotations_in_order(write_record) -> None:
    data_dir = write_record(100, ["N", "A", "V"], sig_names=("MLII", "V5"))

    record = load_record(data_dir, 100)

    assert record.symbols == ("+", "N", "A", "V")
    assert np.all(np.diff(record.ann_samples) > 0)


def test_load_record_raises_when_mlii_absent(write_record) -> None:
    # Arrange: like paced records 102/104, which carry V5 and V2 only
    data_dir = write_record(102, ["N", "N"], sig_names=("V5", "V2"))

    # Act / Assert
    with pytest.raises(LeadNotFoundError, match="102"):
        load_record(data_dir, 102)


def test_download_calls_wfdb_with_mitdb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Arrange
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(records.wfdb, "dl_database", lambda db, dl_dir: calls.append((db, dl_dir)))

    # Act
    download_mitdb(tmp_path / "mitdb")

    # Assert
    assert calls == [("mitdb", str(tmp_path / "mitdb"))]
    assert records.wfdb is wfdb
