"""Model registry: every trainable model by name, with its input kind and batch size."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import keras

from arrhythmia.config import FS, ImageConfig, WindowConfig
from arrhythmia.data.rr import RR_FEATURE_NAMES
from arrhythmia.models.blocks import build_image_cnn
from arrhythmia.models.cnn1d import build_cnn1d, build_cnn1d_rr
from arrhythmia.models.image_cnns import ALEXNET, ALEXNET_WIDE


class ModelKind(Enum):
    """Which input representation a model consumes."""

    IMAGE_2D = "image_2d"
    SIGNAL_1D = "signal_1d"
    SIGNAL_1D_RR = "signal_1d_rr"


@dataclass(frozen=True)
class ModelSpec:
    """How to build a model and feed it."""

    builder: Callable[[int], keras.Model]
    kind: ModelKind
    batch_size: int


class UnknownModelError(ValueError):
    """Raised for a model name that is not in MODEL_REGISTRY."""


def _window_len() -> int:
    before, after = WindowConfig().samples(FS)
    return before + after


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "alexnet": ModelSpec(
        lambda n: build_image_cnn(ALEXNET, ImageConfig().size, n), ModelKind.IMAGE_2D, 64
    ),
    "alexnet_wide": ModelSpec(
        lambda n: build_image_cnn(ALEXNET_WIDE, ImageConfig().size, n), ModelKind.IMAGE_2D, 64
    ),
    "cnn1d": ModelSpec(lambda n: build_cnn1d(_window_len(), n), ModelKind.SIGNAL_1D, 256),
    "cnn1d_rr": ModelSpec(
        lambda n: build_cnn1d_rr(_window_len(), len(RR_FEATURE_NAMES), n),
        ModelKind.SIGNAL_1D_RR,
        256,
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    """Look up a registered model.

    Raises:
        UnknownModelError: If the name is not registered.
    """
    if name not in MODEL_REGISTRY:
        raise UnknownModelError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name]


def build_model(name: str, n_classes: int) -> keras.Model:
    """Build an uncompiled model by registry name."""
    return get_model_spec(name).builder(n_classes)
