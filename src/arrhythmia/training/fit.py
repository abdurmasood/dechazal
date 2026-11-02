"""Seeded, class-weighted training with patient-level leakage checks."""

import math
from dataclasses import dataclass

import keras
import numpy as np

from arrhythmia.data.splits import assert_disjoint_patients


@dataclass(frozen=True)
class TrainConfig:
    """Fixed optimisation settings (not tuned).

    Early stopping watches class-weighted validation loss rather than a macro-F1 callback,
    which would be noisy with ~23 validation F beats.
    """

    epochs: int = 60
    learning_rate: float = 1e-3
    patience: int = 8
    seed: int = 0


@dataclass(frozen=True)
class SplitArrays:
    """Model inputs for one split, every row tagged with its patient.

    Attributes:
        inputs: (main_input,) or (main_input, rr). The main input is beat windows (n, L) or
            images (n, H, W), possibly a memmap; a channel axis is added per batch.
        labels: Class indices, shape (n,).
        record_ids: Patient of each row, shape (n,).
    """

    inputs: tuple[np.ndarray, ...]
    labels: np.ndarray
    record_ids: np.ndarray


@dataclass(frozen=True)
class SplitData:
    """Train, validation and test splits."""

    train: SplitArrays
    val: SplitArrays
    test: SplitArrays


def class_weights(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced class weights n / (k * count_c), as in sklearn's "balanced" mode.

    Raises:
        ValueError: If any class has no training examples.
    """
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=n_classes)[:n_classes]
    missing = np.flatnonzero(counts == 0)
    if missing.size:
        raise ValueError(f"No training examples for class {int(missing[0])}")
    return counts.sum() / (n_classes * counts)


class BeatSequence(keras.utils.PyDataset):
    """Batches (inputs, labels, sample_weights), reading memmaps lazily per batch."""

    def __init__(
        self,
        split: SplitArrays,
        sample_weights: np.ndarray | None,
        batch_size: int,
        shuffle_seed: int | None,
    ) -> None:
        super().__init__()
        self._split = split
        self._weights = sample_weights
        self._batch_size = batch_size
        self._rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
        self._order = np.arange(split.labels.size)
        self._reshuffle()

    def _reshuffle(self) -> None:
        if self._rng is not None:
            self._order = self._rng.permutation(self._split.labels.size)

    def __len__(self) -> int:
        return math.ceil(self._split.labels.size / self._batch_size)

    def __getitem__(self, index: int) -> tuple:
        # Sorted indices keep memmap reads sequential within a batch.
        rows = np.sort(self._order[index * self._batch_size : (index + 1) * self._batch_size])
        main = np.asarray(self._split.inputs[0][rows], dtype=np.float32)[..., None]
        extra = [np.asarray(a[rows], dtype=np.float32) for a in self._split.inputs[1:]]
        x = (main, *extra) if extra else main
        y = self._split.labels[rows]
        weights = self._weights[rows] if self._weights is not None else np.ones(rows.size)
        return x, y, weights.astype(np.float32)

    def on_epoch_end(self) -> None:
        self._reshuffle()


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and the Keras backend. Call before building the model.

    Weights are initialised when the model is built, so seeding only inside fit would leave
    initialisation dependent on whatever ran earlier in the process.
    """
    keras.utils.set_random_seed(seed)


def fit_model(
    model: keras.Model, data: SplitData, cfg: TrainConfig, batch_size: int
) -> keras.callbacks.History:
    """Compile and train with class weights and early stopping on weighted val loss.

    Call set_seed(cfg.seed) before building `model`; cfg.seed also seeds batch shuffling.

    Raises:
        PatientOverlapError: If any patient appears in more than one split. Checked before
            any training happens.
    """
    assert_disjoint_patients(
        {name: getattr(data, name).record_ids for name in ("train", "val", "test")}
    )
    n_classes = int(model.output_shape[-1])
    weights = class_weights(data.train.labels, n_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        weighted_metrics=["accuracy"],
    )
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience, restore_best_weights=True
    )
    return model.fit(
        BeatSequence(data.train, weights[data.train.labels], batch_size, cfg.seed),
        validation_data=BeatSequence(data.val, weights[data.val.labels], batch_size, None),
        epochs=cfg.epochs,
        callbacks=[stop],
        verbose=2,
    )


def predict_classes(model: keras.Model, split: SplitArrays, batch_size: int) -> np.ndarray:
    """Predicted class index for every row of a split, in row order."""
    probabilities = model.predict(BeatSequence(split, None, batch_size, None), verbose=0)
    return probabilities.argmax(axis=1)
