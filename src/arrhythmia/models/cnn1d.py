"""Small 1D CNNs on the filtered raw beat window, optionally with RR features.

One fixed architecture, not searched: three Conv1D stages (32/64/128 filters, kernel 7,
~20 ms at 360 Hz) with BatchNorm and pooling, then global average pooling.
"""

import keras
from keras import layers

CONV_FILTERS: tuple[int, ...] = (32, 64, 128)
KERNEL_SIZE = 7
DROPOUT = 0.3
RR_HIDDEN_UNITS = 16


def _signal_trunk(signal: keras.KerasTensor) -> keras.KerasTensor:
    x = signal
    for filters in CONV_FILTERS:
        x = layers.Conv1D(filters, KERNEL_SIZE, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling1D(2)(x)
    return layers.GlobalAveragePooling1D()(x)


def build_cnn1d(window_len: int, n_classes: int) -> keras.Model:
    """Waveform-only 1D CNN: the like-for-like comparison against the 2D image models."""
    signal = keras.Input((window_len, 1), name="signal")
    x = layers.Dropout(DROPOUT)(_signal_trunk(signal))
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(signal, outputs, name="cnn1d")


def build_cnn1d_rr(window_len: int, n_rr: int, n_classes: int) -> keras.Model:
    """1D CNN plus an RR-feature branch concatenated after global pooling."""
    signal = keras.Input((window_len, 1), name="signal")
    rr = keras.Input((n_rr,), name="rr")
    rr_features = layers.Dense(RR_HIDDEN_UNITS, activation="relu")(rr)
    x = layers.Concatenate()([_signal_trunk(signal), rr_features])
    x = layers.Dropout(DROPOUT)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model([signal, rr], outputs, name="cnn1d_rr")
