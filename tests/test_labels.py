"""Tests for the MIT-BIH symbol to AAMI class mapping."""

import pytest

from arrhythmia.data.labels import TRAINING_CLASSES, AamiClass, UnknownSymbolError, to_aami

EXPECTED_BEAT_MAPPING = {
    "N": AamiClass.N,
    "L": AamiClass.N,
    "R": AamiClass.N,
    "e": AamiClass.N,
    "j": AamiClass.N,
    "A": AamiClass.S,
    "a": AamiClass.S,
    "J": AamiClass.S,
    "S": AamiClass.S,
    "V": AamiClass.V,
    "E": AamiClass.V,
    "F": AamiClass.F,
    "/": AamiClass.Q,
    "f": AamiClass.Q,
    "Q": AamiClass.Q,
}


@pytest.mark.parametrize(("symbol", "expected"), EXPECTED_BEAT_MAPPING.items())
def test_symbol_maps_to_aami_class(symbol: str, expected: AamiClass) -> None:
    assert to_aami(symbol) is expected


@pytest.mark.parametrize("symbol", ["+", "~", "|", '"', "x", "!", "[", "]"])
def test_non_beat_symbol_returns_none(symbol: str) -> None:
    assert to_aami(symbol) is None


def test_unknown_symbol_raises() -> None:
    with pytest.raises(UnknownSymbolError, match="'Z'"):
        to_aami("Z")


def test_training_classes_are_n_s_v_f_with_contiguous_indices() -> None:
    assert [int(c) for c in TRAINING_CLASSES] == [0, 1, 2, 3]
    assert AamiClass.Q not in TRAINING_CLASSES
