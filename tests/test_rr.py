"""Tests for RR-interval features."""

import numpy as np
import pytest

from arrhythmia.data.rr import RR_FEATURE_NAMES, rr_features

FS = 360


def test_rr_features_from_known_sample_positions() -> None:
    # Arrange: intervals of 1.0 s, 1.0 s, 1.5 s -> record mean RR = 3.5 / 3 s
    beats = np.array([0, 360, 720, 1260])
    mean_rr = 3.5 / 3

    # Act
    features = rr_features(beats, FS)

    # Assert
    assert features.shape == (4, len(RR_FEATURE_NAMES))
    pre, post = features[:, 0] * mean_rr, features[:, 1] * mean_rr
    np.testing.assert_allclose(pre, [1.0, 1.0, 1.0, 1.5], rtol=1e-6)
    np.testing.assert_allclose(post, [1.0, 1.0, 1.5, 1.5], rtol=1e-6)


def test_local_rr_is_mean_of_up_to_10_preceding_pre_rr() -> None:
    beats = np.array([0, 360, 720, 1260])
    mean_rr = 3.5 / 3

    features = rr_features(beats, FS)

    local = features[:, 2] * mean_rr
    np.testing.assert_allclose(local, [1.0, 1.0, 1.0, (1.0 + 1.0 + 1.0 + 1.5) / 4], rtol=1e-6)
    np.testing.assert_allclose(features[:, 3], [1.0, 1.0, 1.0, 1.5 / 1.125], rtol=1e-6)


def test_rr_normalised_by_record_mean() -> None:
    # Arrange: same rhythm at half the heart rate
    beats = np.array([0, 360, 720, 1260])

    # Act
    normal_rate = rr_features(beats, FS)
    half_rate = rr_features(beats * 2, FS)

    # Assert: patient heart-rate level is removed
    np.testing.assert_allclose(normal_rate, half_rate, rtol=1e-6)


def test_first_beat_rr_uses_post_rr_fallback() -> None:
    beats = np.array([0, 500, 900])

    features = rr_features(beats, FS)

    assert features[0, 0] == pytest.approx(features[0, 1])


def test_rr_features_require_two_beats() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        rr_features(np.array([100]), FS)
