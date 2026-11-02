"""Tests for rendering beat windows as 2D images."""

from pathlib import Path

import numpy as np

from arrhythmia.config import ImageConfig
from arrhythmia.data.images import load_or_render_images, render_beat_image

WINDOW = np.sin(np.linspace(0, 2 * np.pi, 252)).astype(np.float32)


def test_render_returns_224_square_uint8() -> None:
    image = render_beat_image(WINDOW, ImageConfig())

    assert image.shape == (224, 224)
    assert image.dtype == np.uint8
    assert image.min() == 0  # line pixels
    assert image.max() == 255  # background


def test_render_scales_each_beat_to_full_height() -> None:
    # Arrange: same shape, 10x amplitude
    small, large = WINDOW * 0.1, WINDOW * 1.0

    # Act / Assert: per-beat min-max scaling discards absolute amplitude
    np.testing.assert_array_equal(
        render_beat_image(small, ImageConfig()), render_beat_image(large, ImageConfig())
    )


def test_render_flat_signal_draws_midline_without_error() -> None:
    image = render_beat_image(np.zeros(252, dtype=np.float32), ImageConfig())

    dark_rows = np.flatnonzero((image < 128).any(axis=1))
    assert dark_rows.size > 0
    assert abs(dark_rows.mean() - 111.5) < 2


def test_render_is_deterministic() -> None:
    np.testing.assert_array_equal(
        render_beat_image(WINDOW, ImageConfig()), render_beat_image(WINDOW, ImageConfig())
    )


def test_memmap_row_count_equals_beat_count(tmp_path: Path) -> None:
    # Arrange
    signals = np.stack([WINDOW, -WINDOW, WINDOW * 2])

    # Act
    images = load_or_render_images(signals, tmp_path / "img.u8", ImageConfig())

    # Assert
    assert images.shape == (3, 224, 224)
    np.testing.assert_array_equal(images[1], render_beat_image(-WINDOW, ImageConfig()))


def test_existing_memmap_is_reused(tmp_path: Path) -> None:
    signals = np.stack([WINDOW, -WINDOW])
    path = tmp_path / "img.u8"
    load_or_render_images(signals, path, ImageConfig())
    mtime = path.stat().st_mtime_ns

    reopened = load_or_render_images(signals, path, ImageConfig())

    assert path.stat().st_mtime_ns == mtime
    assert reopened.shape == (2, 224, 224)
