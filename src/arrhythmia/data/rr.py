"""RR-interval (rhythm) features per beat.

S beats are often morphologically normal and stand out mainly by prematurity, so the
cnn1d_rr model receives these alongside the waveform. Post-RR uses the next beat, which
makes the classifier offline (it needs one beat of look-ahead).
"""

import numpy as np

RR_FEATURE_NAMES: tuple[str, ...] = ("pre_rr", "post_rr", "local_rr", "pre_over_local")
LOCAL_WINDOW_BEATS = 10


def rr_features(beat_samples: np.ndarray, fs: float) -> np.ndarray:
    """Compute RR features for every beat of one record.

    pre/post/local RR are divided by the record's mean RR to remove the patient's baseline
    heart rate; pre_over_local is already unitless.

    Args:
        beat_samples: Sorted R-peak sample indices of all beats in the record (all AAMI
            classes, including Q, so intervals are not distorted by dropped beats).
        fs: Sampling frequency in Hz.

    Returns:
        Array of shape (n_beats, 4) in RR_FEATURE_NAMES order, float32.

    Raises:
        ValueError: If fewer than two beats are given.
    """
    if beat_samples.size < 2:
        raise ValueError(f"RR features need at least 2 beats, got {beat_samples.size}")
    intervals = np.diff(beat_samples.astype(np.float64)) / fs
    # The first beat has no predecessor and the last no successor: borrow the neighbour.
    pre = np.concatenate([intervals[:1], intervals])
    post = np.concatenate([intervals, intervals[-1:]])
    cumulative = np.concatenate([[0.0], np.cumsum(pre)])
    idx = np.arange(pre.size)
    start = np.maximum(0, idx - LOCAL_WINDOW_BEATS + 1)
    local = (cumulative[idx + 1] - cumulative[start]) / (idx + 1 - start)
    mean_rr = intervals.mean()
    features = np.column_stack([pre / mean_rr, post / mean_rr, local / mean_rr, pre / local])
    return features.astype(np.float32)
