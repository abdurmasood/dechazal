"""Shared layer blocks for the 2019 image CNNs (AlexNet-style and NovelNet).

Both 2019 models used the same pattern, Conv -> ReLU -> [MaxPool] -> BatchNorm for
convolutions and Dense -> ReLU -> Dropout -> BatchNorm for the head. They differ only in
layer sizes, so each one is a pair of spec tuples fed to one builder.
"""

from dataclasses import dataclass

import keras
from keras import layers


@dataclass(frozen=True)
class ConvSpec:
    """One convolution stage. Padding is 'valid' throughout, as in the 2019 code."""

    filters: int
    kernel: int
    strides: int = 1
    pool: bool = False


@dataclass(frozen=True)
class DenseSpec:
    """One fully connected stage of the classifier head."""

    units: int
    dropout: float


def conv_block(x: keras.KerasTensor, spec: ConvSpec) -> keras.KerasTensor:
    """Conv2D -> ReLU -> optional 2x2 MaxPool -> BatchNorm."""
    x = layers.Conv2D(spec.filters, spec.kernel, strides=spec.strides, padding="valid")(x)
    x = layers.Activation("relu")(x)
    if spec.pool:
        x = layers.MaxPooling2D(pool_size=2, strides=2, padding="valid")(x)
    return layers.BatchNormalization()(x)


def dense_block(x: keras.KerasTensor, spec: DenseSpec) -> keras.KerasTensor:
    """Dense -> ReLU -> Dropout -> BatchNorm."""
    x = layers.Dense(spec.units)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(spec.dropout)(x)
    return layers.BatchNormalization()(x)


@dataclass(frozen=True)
class ImageCnnSpec:
    """Full layer specification of one 2019-style image CNN."""

    name: str
    convs: tuple[ConvSpec, ...]
    denses: tuple[DenseSpec, ...]


def build_image_cnn(spec: ImageCnnSpec, image_size: int, n_classes: int) -> keras.Model:
    """Build a single-channel image CNN from its spec.

    Args:
        spec: Conv and dense stages.
        image_size: Side length of the square grayscale input.
        n_classes: Number of output classes.

    Returns:
        Uncompiled functional Keras model taking uint8-range pixels.
    """
    inputs = keras.Input((image_size, image_size, 1), name="image")
    x = layers.Rescaling(1.0 / 255)(inputs)
    for conv in spec.convs:
        x = conv_block(x, conv)
    x = layers.Flatten()(x)
    for dense in spec.denses:
        x = dense_block(x, dense)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name=spec.name)
