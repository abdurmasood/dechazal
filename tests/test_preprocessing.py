"""Tests for ECG filtering and beat windowing."""

import numpy as np

from arrhythmia.config import FS, FilterConfig, WindowConfig
from arrhythmia.data.preprocessing import extract_windows, lowpass, remove_baseline

DURATION_S = 30
T = np.arange(DURATION_S * FS) / FS


def _amplitude(signal: np.ndarray) -> float:
    """Peak amplitude away from the edges, where filters have transients."""
    core = signal[FS:-FS]
    return float((core.max() - core.min()) / 2)


def test_remove_baseline_suppresses_slow_drift() -> None:
    # Arrange: 0.2 Hz respiration-like wander
    drift = 0.5 * np.sin(2 * np.pi * 0.2 * T)

    # Act
    cleaned = remove_baseline(drift, FS, FilterConfig())

    # Assert
    assert _amplitude(cleaned) < 0.1 * _amplitude(drift)


def test_remove_baseline_keeps_narrow_qrs_peak() -> None:
    # Arrange: a 20 ms wide spike is far narrower than the 200 ms median window
    signal = np.zeros_like(T)
    r = 10 * FS
    signal[r - 4 : r + 4] = 1.0

    # Act
    cleaned = remove_baseline(signal, FS, FilterConfig())

    # Assert
    assert cleaned[r] > 0.95


def test_lowpass_attenuates_60hz() -> None:
    mains = np.sin(2 * np.pi * 60 * T)

    filtered = lowpass(mains, FS, FilterConfig())

    assert _amplitude(filtered) < 0.05


def test_lowpass_preserves_5hz() -> None:
    qrs_band = np.sin(2 * np.pi * 5 * T)

    filtered = lowpass(qrs_band, FS, FilterConfig())

    assert _amplitude(filtered) > 0.95


def test_extract_windows_centres_r_peak_at_index_90() -> None:
    # Arrange
    signal = np.zeros(10 * FS)
    r_samples = np.array([2 * FS, 5 * FS])
    signal[r_samples] = 1.0

    # Act
    windows, kept = extract_windows(signal, r_samples, FS, WindowConfig())

    # Assert
    assert windows.shape == (2, 252)
    assert kept.tolist() == [True, True]
    assert np.all(windows.argmax(axis=1) == 90)


def test_extract_windows_drops_out_of_bounds_beats() -> None:
    # Arrange: first beat too close to the start, last too close to the end
    signal = np.zeros(10 * FS)
    r_samples = np.array([50, 5 * FS, 10 * FS - 100])

    # Act
    windows, kept = extract_windows(signal, r_samples, FS, WindowConfig())

    # Assert
    assert kept.tolist() == [False, True, False]
    assert windows.shape == (1, 252)
