"""ECG filtering and fixed-length beat windowing."""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

from arrhythmia.config import FilterConfig, WindowConfig


def _odd_length(seconds: float, fs: float) -> int:
    length = round(seconds * fs)
    return length if length % 2 == 1 else length + 1


def remove_baseline(signal: np.ndarray, fs: float, cfg: FilterConfig) -> np.ndarray:
    """Subtract a cascaded-median baseline estimate (de Chazal et al., 2004).

    The first window removes QRS complexes and P waves, the second removes T waves; what is
    left is the baseline wander.

    Args:
        signal: Raw ECG in mV.
        fs: Sampling frequency in Hz.
        cfg: Filter settings.

    Returns:
        Baseline-corrected signal, same shape as the input.
    """
    baseline = signal
    for window_s in cfg.baseline_windows_s:
        baseline = median_filter(baseline, size=_odd_length(window_s, fs), mode="nearest")
    return signal - baseline


def lowpass(signal: np.ndarray, fs: float, cfg: FilterConfig) -> np.ndarray:
    """Zero-phase Butterworth low-pass to suppress mains and muscle noise.

    Args:
        signal: ECG in mV.
        fs: Sampling frequency in Hz.
        cfg: Filter settings.

    Returns:
        Filtered signal, same shape as the input.
    """
    sos = butter(cfg.lowpass_order, cfg.lowpass_hz, fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def filter_signal(signal: np.ndarray, fs: float, cfg: FilterConfig) -> np.ndarray:
    """Baseline removal followed by low-pass, applied to a whole record before windowing."""
    return lowpass(remove_baseline(signal, fs, cfg), fs, cfg)


def extract_windows(
    signal: np.ndarray, r_samples: np.ndarray, fs: float, cfg: WindowConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Cut a fixed window around each R peak, dropping beats that run off the record.

    Args:
        signal: Filtered ECG.
        r_samples: Annotated R-peak sample indices.
        fs: Sampling frequency in Hz.
        cfg: Window extent.

    Returns:
        (windows, kept): windows has shape (n_kept, before + after); kept is a boolean mask
        over r_samples marking which beats produced a window.
    """
    before, after = cfg.samples(fs)
    r_samples = np.asarray(r_samples, dtype=np.int64)
    kept = (r_samples - before >= 0) & (r_samples + after <= signal.size)
    offsets = np.arange(-before, after)
    windows = signal[r_samples[kept, None] + offsets[None, :]]
    return windows, kept
