"""Characterization tests documenting the 2019 label bug in cnn_model.trainAndTestSplit.

These tests PASS while the bug exists. They are evidence for the README, not a spec:
the legacy code is frozen and replaced by src/arrhythmia in Phase B.

Mechanism (scripts/cnn_model.py:156-181): `classes_to_check = CLASSES_TO_CHECK` aliases
the module-level list, `.remove()` drops each class once it reaches its cap, and labels
are computed with `classes_to_check.index(...)` against that shrinking list. Rows arrive
grouped by class folder, so each class is labelled while it is at index 0.
"""

import numpy as np
import pandas as pd
import pytest

import cnn_model

LEGACY_CLASSES = ["L", "N", "V", "A", "R"]
ROWS_PER_CLASS = 2


@pytest.fixture
def legacy_split_inputs(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Fresh global class list, a cap of 2 per class, and 2 rows per class in L,N,V,A,R order."""
    monkeypatch.setattr(cnn_model, "CLASSES_TO_CHECK", list(LEGACY_CLASSES))
    monkeypatch.setattr(cnn_model, "IMAGES_TO_TRAIN", ROWS_PER_CLASS)
    rows = [
        {"Signal ID": f"{cls}_{i}", "Signal": np.zeros((2, 2, 3)), "Type": cls}
        for cls in LEGACY_CLASSES
        for i in range(ROWS_PER_CLASS)
    ]
    return pd.DataFrame(rows)


def test_legacy_split_assigns_same_label_to_every_class(legacy_split_inputs: pd.DataFrame) -> None:
    # Arrange
    df = legacy_split_inputs

    # Act
    _, _, y_train, y_test = cnn_model.trainAndTestSplit(df, 0.2)

    # Assert: five distinct classes in, a single label out
    labels = np.concatenate([y_train.argmax(axis=1), y_test.argmax(axis=1)])
    assert len(labels) == len(LEGACY_CLASSES) * ROWS_PER_CLASS
    assert set(labels.tolist()) == {0}


def test_legacy_split_empties_global_class_list(legacy_split_inputs: pd.DataFrame) -> None:
    # Arrange
    df = legacy_split_inputs

    # Act
    cnn_model.trainAndTestSplit(df, 0.2)

    # Assert: the module-level constant was mutated through the alias
    assert cnn_model.CLASSES_TO_CHECK == []
