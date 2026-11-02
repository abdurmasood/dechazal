"""Tests for the model registry and architectures."""

import pytest

from arrhythmia.models import MODEL_REGISTRY, ModelKind, UnknownModelError, build_model

N_CLASSES = 4


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_build_model_output_matches_class_count(name: str) -> None:
    model = build_model(name, N_CLASSES)

    assert model.output_shape == (None, N_CLASSES)


def test_unknown_model_name_raises() -> None:
    with pytest.raises(UnknownModelError, match="resnet"):
        build_model("resnet", N_CLASSES)


@pytest.mark.parametrize("name", ["alexnet", "alexnet_wide"])
def test_2d_models_take_single_channel_224_images(name: str) -> None:
    model = build_model(name, N_CLASSES)

    assert MODEL_REGISTRY[name].kind is ModelKind.IMAGE_2D
    assert model.input_shape == (None, 224, 224, 1)


def test_alexnet_wide_uses_pre_aff2afe_conv_filters() -> None:
    # aff2afe (2019-04-04) added a sixth conv that cannot run on the 2x2 feature map.
    model = build_model("alexnet_wide", N_CLASSES)

    conv_filters = [layer.filters for layer in model.layers if layer.__class__.__name__ == "Conv2D"]
    assert conv_filters == [96, 256, 384, 434, 500]


def test_cnn1d_param_count_under_200k() -> None:
    model = build_model("cnn1d", N_CLASSES)

    assert model.count_params() < 200_000
    assert model.input_shape == (None, 252, 1)


def test_cnn1d_rr_takes_signal_and_rr_inputs() -> None:
    model = build_model("cnn1d_rr", N_CLASSES)

    assert [tuple(i.shape) for i in model.inputs] == [(None, 252, 1), (None, 4)]
