"""Rendering beat windows as grayscale line-plot images for the 2D CNN baseline.

This reproduces the 2019 representation (a plotted beat with auto-scaled axes) without
matplotlib: PIL draws straight into a uint8 array, so ~100k images take minutes, not hours.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from arrhythmia.config import ImageConfig

BACKGROUND = 255
INK = 0


def render_beat_image(window: np.ndarray, cfg: ImageConfig) -> np.ndarray:
    """Draw one beat as a dark line on a white square.

    The vertical axis is min-max scaled per beat, like matplotlib's autoscale, so absolute
    amplitude is lost. That loss is part of what the 1D-vs-2D comparison measures.

    Args:
        window: One beat window (1D).
        cfg: Image size and line width.

    Returns:
        uint8 array of shape (cfg.size, cfg.size).
    """
    top = cfg.size - 1
    low, high = float(window.min()), float(window.max())
    flat = high - low == 0
    y = np.full(window.size, top / 2) if flat else (high - window) / (high - low) * top
    x = np.linspace(0, top, window.size)
    image = Image.new("L", (cfg.size, cfg.size), BACKGROUND)
    ImageDraw.Draw(image).line(
        list(zip(x.tolist(), y.tolist(), strict=True)), fill=INK, width=cfg.line_width
    )
    return np.asarray(image, dtype=np.uint8)


def load_or_render_images(signals: np.ndarray, path: Path, cfg: ImageConfig) -> np.memmap:
    """Return a read-only memmap of rendered images, rendering it on first use.

    Args:
        signals: Beat windows, shape (n, window_len).
        path: Memmap file location; reused if it already holds n images of this size.
        cfg: Image settings.

    Returns:
        uint8 memmap of shape (n, cfg.size, cfg.size).
    """
    shape = (signals.shape[0], cfg.size, cfg.size)
    path = Path(path)
    if path.exists() and path.stat().st_size == int(np.prod(shape)):
        return np.memmap(path, dtype=np.uint8, mode="r", shape=shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = np.memmap(path, dtype=np.uint8, mode="w+", shape=shape)
    for i, window in enumerate(signals):
        images[i] = render_beat_image(window, cfg)
    images.flush()
    del images
    return np.memmap(path, dtype=np.uint8, mode="r", shape=shape)
