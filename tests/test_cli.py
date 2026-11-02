"""End-to-end smoke test of the CLI on synthetic WFDB records (no network, CPU)."""

import json
from pathlib import Path

import pytest

from arrhythmia.cli import main
from arrhythmia.data.splits import DS1, DS2

# Every record carries all four training classes plus one Q beat.
SYMBOLS = ["N", "N", "A", "V", "F", "N", "Q", "N", "A", "V", "F", "N"]


@pytest.fixture
def synthetic_mitdb(write_record, tmp_path: Path) -> Path:
    data_dir = tmp_path / "mitdb"
    for record_id in sorted(DS1 | DS2):
        write_record(record_id, SYMBOLS, sig_names=("MLII", "V5"), directory=data_dir)
    return data_dir


def test_cli_smoke_end_to_end(
    synthetic_mitdb: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Arrange
    dirs = ["--data-dir", str(synthetic_mitdb), "--cache-dir", str(tmp_path / "cache")]
    runs = tmp_path / "runs"

    # Act
    assert main(["prepare", *dirs]) == 0
    assert main(["train", "--model", "cnn1d", "--seeds", "0", "--epochs", "1", *dirs,
                 "--runs-dir", str(runs)]) == 0  # fmt: skip
    assert main(["report", str(runs)]) == 0

    # Assert
    seed_dir = runs / "cnn1d" / "seed0"
    for artifact in ("model.keras", "metrics.json", "confusion.csv", "config.json"):
        assert (seed_dir / artifact).exists(), artifact
    summary = json.loads((runs / "dataset.json").read_text())
    assert summary["test"]["excluded_q"] == len(DS2)
    output = capsys.readouterr().out
    assert "| cnn1d | 1 |" in output
    assert f"Q excluded: {len(DS1)} (DS1), {len(DS2)} (DS2)" in output


def test_cli_rejects_unknown_model(synthetic_mitdb: Path, tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["train", "--model", "resnet", "--data-dir", str(synthetic_mitdb)])
