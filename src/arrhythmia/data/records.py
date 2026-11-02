"""Loading MIT-BIH records from disk and downloading them from PhysioNet."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

LEAD = "MLII"


class LeadNotFoundError(ValueError):
    """Raised when a record has no MLII lead (e.g. paced records 102 and 104)."""


@dataclass(frozen=True)
class Record:
    """One MIT-BIH record: the MLII signal plus every reference annotation.

    Attributes:
        record_id: Numeric record name, e.g. 100.
        signal: MLII lead in physical units (mV).
        fs: Sampling frequency in Hz.
        ann_samples: Sample index of every annotation, beats and non-beats alike.
        symbols: WFDB symbol of every annotation, aligned with ann_samples.
    """

    record_id: int
    signal: np.ndarray
    fs: float
    ann_samples: np.ndarray
    symbols: tuple[str, ...]


def load_record(data_dir: Path, record_id: int) -> Record:
    """Read the MLII lead and reference annotations of one record.

    The lead is selected by name: channel 0 is not MLII in records 102, 104 and 114.

    Args:
        data_dir: Directory holding <record_id>.hea/.dat/.atr.
        record_id: Numeric record name.

    Returns:
        The loaded record.

    Raises:
        LeadNotFoundError: If the header lists no MLII signal.
    """
    path = str(Path(data_dir) / str(record_id))
    header = wfdb.rdheader(path)
    if LEAD not in header.sig_name:
        raise LeadNotFoundError(f"Record {record_id} has no {LEAD} lead (found {header.sig_name})")
    wfdb_record = wfdb.rdrecord(path, channel_names=[LEAD])
    annotation = wfdb.rdann(path, "atr")
    return Record(
        record_id=record_id,
        signal=wfdb_record.p_signal[:, 0].astype(np.float64),
        fs=float(wfdb_record.fs),
        ann_samples=np.asarray(annotation.sample, dtype=np.int64),
        symbols=tuple(annotation.symbol),
    )


def download_mitdb(data_dir: Path) -> None:
    """Download the MIT-BIH Arrhythmia Database (~100 MB) from PhysioNet.

    Args:
        data_dir: Destination directory; created if missing.
    """
    wfdb.dl_database("mitdb", dl_dir=str(data_dir))
