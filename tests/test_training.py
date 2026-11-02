"""Tests for class weighting, batching and model fitting."""

import numpy as np
import pytest

from arrhythmia.data.splits import PatientOverlapError
from arrhythmia.models import build_model
from arrhythmia.training.fit import (
    BeatSequence,
    SplitArrays,
    SplitData,
    TrainConfig,
    class_weights,
    fit_model,
    predict_classes,
    set_seed,
)

WINDOW_LEN = 252
N_CLASSES = 4


def _separable_split(record_id: int, n_per_class: int, seed: int) -> SplitArrays:
    """Each class is an R-centred bump with its own polarity and width, plus noise.

    Real beats are aligned on the R peak and differ in morphology, not position (and the
    1D model's global average pooling is position-invariant), so the synthetic ones are too.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW_LEN)
    shapes = ((1.0, 4), (-1.0, 4), (1.0, 15), (-1.0, 15))  # (amplitude, width in samples)
    labels = np.repeat(np.arange(N_CLASSES), n_per_class).astype(np.int8)
    templates = np.stack([a * np.exp(-0.5 * ((t - 90) / w) ** 2) for a, w in shapes])
    signals = templates[labels] + 0.1 * rng.standard_normal((labels.size, WINDOW_LEN))
    record_ids = np.full(labels.size, record_id, dtype=np.int16)
    return SplitArrays(inputs=(signals.astype(np.float32),), labels=labels, record_ids=record_ids)


def test_class_weights_inverse_to_frequency() -> None:
    labels = np.array([0] * 8 + [1] * 2)

    weights = class_weights(labels, 2)

    np.testing.assert_allclose(weights, [10 / (2 * 8), 10 / (2 * 2)])


def test_class_weights_raise_when_training_class_missing() -> None:
    with pytest.raises(ValueError, match="class 2"):
        class_weights(np.array([0, 1, 1]), 3)


def test_sequence_appends_channel_axis_and_yields_weights() -> None:
    # Arrange
    split = _separable_split(101, n_per_class=5, seed=0)
    weights = np.arange(20, dtype=np.float32)

    # Act
    x, y, sw = BeatSequence(split, weights, batch_size=8, shuffle_seed=None)[0]

    # Assert
    assert x.shape == (8, WINDOW_LEN, 1)
    np.testing.assert_array_equal(y, split.labels[:8])
    np.testing.assert_array_equal(sw, weights[:8])


def test_sequence_yields_signal_and_rr_inputs_when_rr_present() -> None:
    base = _separable_split(101, n_per_class=2, seed=0)
    rr = np.ones((8, 4), dtype=np.float32)
    split = SplitArrays(inputs=(base.inputs[0], rr), labels=base.labels, record_ids=base.record_ids)

    (signal, rr_batch), _, _ = BeatSequence(split, None, batch_size=4, shuffle_seed=None)[0]

    assert signal.shape == (4, WINDOW_LEN, 1)
    assert rr_batch.shape == (4, 4)


def test_sequence_shuffle_is_deterministic_per_seed() -> None:
    split = _separable_split(101, n_per_class=10, seed=0)

    # Act: first batch of 10 out of 40 rows (the unshuffled first batch is all class 0)
    first = BeatSequence(split, None, batch_size=10, shuffle_seed=3)[0][1]
    second = BeatSequence(split, None, batch_size=10, shuffle_seed=3)[0][1]

    # Assert
    np.testing.assert_array_equal(first, second)
    assert len(set(first.tolist())) > 1


def test_fit_learns_separable_synthetic_classes() -> None:
    # Arrange
    data = SplitData(
        train=_separable_split(101, n_per_class=64, seed=0),
        val=_separable_split(114, n_per_class=16, seed=1),
        test=_separable_split(100, n_per_class=16, seed=2),
    )
    cfg = TrainConfig(epochs=4, patience=4, seed=0)
    set_seed(cfg.seed)
    model = build_model("cnn1d", N_CLASSES)

    # Act
    fit_model(model, data, cfg, batch_size=32)
    predicted = predict_classes(model, data.test, batch_size=64)

    # Assert
    assert np.mean(predicted == data.test.labels) > 0.9


def test_fit_refuses_overlapping_patients() -> None:
    # Arrange: record 100 is in both train and test
    data = SplitData(
        train=_separable_split(100, n_per_class=4, seed=0),
        val=_separable_split(114, n_per_class=4, seed=1),
        test=_separable_split(100, n_per_class=4, seed=2),
    )

    # Act / Assert
    with pytest.raises(PatientOverlapError, match="100"):
        fit_model(build_model("cnn1d", N_CLASSES), data, TrainConfig(epochs=1), batch_size=8)


def test_2d_model_trains_on_memmap_images(tmp_path) -> None:
    # Arrange: render 8 beats per split to memmaps, as the alexnet pipeline does
    from arrhythmia.config import ImageConfig
    from arrhythmia.data.images import load_or_render_images

    def _image_split(record_id: int, seed: int) -> SplitArrays:
        beats = _separable_split(record_id, n_per_class=2, seed=seed)
        images = load_or_render_images(beats.inputs[0], tmp_path / f"{record_id}.u8", ImageConfig())
        return SplitArrays(inputs=(images,), labels=beats.labels, record_ids=beats.record_ids)

    data = SplitData(
        train=_image_split(101, 0), val=_image_split(114, 1), test=_image_split(100, 2)
    )
    model = build_model("alexnet", N_CLASSES)

    # Act
    history = fit_model(model, data, TrainConfig(epochs=1), batch_size=8)
    predicted = predict_classes(model, data.test, batch_size=8)

    # Assert
    assert len(history.history["loss"]) == 1
    assert predicted.shape == (8,)


def test_same_seed_gives_identical_predictions() -> None:
    # Arrange: weight initialisation must be covered by the seed, not just shuffling
    data = SplitData(
        train=_separable_split(101, n_per_class=16, seed=0),
        val=_separable_split(114, n_per_class=4, seed=1),
        test=_separable_split(100, n_per_class=8, seed=2),
    )
    cfg = TrainConfig(epochs=2, seed=7)

    def _train_and_predict() -> np.ndarray:
        set_seed(cfg.seed)
        model = build_model("cnn1d", N_CLASSES)
        fit_model(model, data, cfg, batch_size=16)
        return model.predict(BeatSequence(data.test, None, 16, None), verbose=0)

    # Act
    first = _train_and_predict()
    np.random.default_rng().standard_normal(10)  # disturb global state between runs
    second = _train_and_predict()

    # Assert
    np.testing.assert_allclose(first, second, rtol=1e-5)
