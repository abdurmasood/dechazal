"""Shared pytest configuration and synthetic WFDB fixtures."""

import os

# Keras picks its backend at import time; pin it before any test module imports keras.
os.environ.setdefault("KERAS_BACKEND", "jax")

from collections.abc import Callable, Sequence  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import wfdb  # noqa: E402

FS = 360
RR_SECONDS = 0.8
LEAD_IN_SECONDS = 1.0

# Per-symbol (amplitude mV, width s) of a Gaussian "QRS". Distinct shapes make the classes
# learnable by a tiny model in a couple of epochs.
SYNTH_SHAPES: dict[str, tuple[float, float]] = {
    "N": (1.0, 0.010),
    "A": (0.5, 0.010),
    "V": (-1.2, 0.040),
    "F": (0.8, 0.030),
    "Q": (0.3, 0.020),
}

WriteRecord = Callable[..., Path]


def synth_mlii(symbols: Sequence[str], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Build an MLII-like trace with one Gaussian per beat symbol.

    Returns:
        The signal in mV and the R-peak sample index of each beat.
    """
    rr = int(RR_SECONDS * FS)
    r_samples = int(LEAD_IN_SECONDS * FS) + rr * np.arange(len(symbols))
    t = np.arange(r_samples[-1] + int(LEAD_IN_SECONDS * FS))
    signal = 0.02 * rng.standard_normal(t.size)
    for r, symbol in zip(r_samples, symbols, strict=True):
        amplitude, width = SYNTH_SHAPES[symbol]
        signal += amplitude * np.exp(-0.5 * ((t - r) / (width * FS)) ** 2)
    return signal, r_samples


@pytest.fixture
def write_record(tmp_path: Path) -> WriteRecord:
    """Return a factory that writes a real WFDB record (.hea/.dat/.atr) into tmp_path."""

    def _write(
        record_id: int,
        symbols: Sequence[str],
        sig_names: tuple[str, str] = ("V5", "MLII"),
        directory: Path | None = None,
    ) -> Path:
        out_dir = directory or tmp_path
        out_dir.mkdir(parents=True, exist_ok=True)
        mlii, r_samples = synth_mlii(symbols, np.random.default_rng(record_id))
        other = np.full_like(mlii, 0.5)
        lead_index = sig_names.index("MLII") if "MLII" in sig_names else 1
        channels = [other, other]
        channels[lead_index] = mlii
        wfdb.wrsamp(
            str(record_id),
            fs=FS,
            units=["mV", "mV"],
            sig_name=list(sig_names),
            p_signal=np.column_stack(channels),
            fmt=["16", "16"],
            write_dir=str(out_dir),
        )
        # A leading rhythm annotation ('+') mimics MIT-BIH, where non-beat events are interleaved.
        ann_samples = np.concatenate([[1], r_samples])
        wfdb.wrann(str(record_id), "atr", ann_samples, ["+", *symbols], write_dir=str(out_dir))
        return out_dir

    return _write
