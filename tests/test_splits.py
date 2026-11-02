"""Tests for the de Chazal inter-patient split."""

import pytest

from arrhythmia.data.splits import (
    DS1,
    DS2,
    PACED,
    TRAIN_RECORDS,
    VAL_RECORDS,
    PatientOverlapError,
    assert_disjoint_patients,
)

# All 48 MIT-BIH Arrhythmia Database record IDs, as found in the .hea files.
ALL_MITDB_RECORDS = frozenset(
    {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113, 114, 115, 116}
    | {117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203, 205, 207, 208, 209, 210}
    | {212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234}
)


def test_ds1_and_ds2_have_22_disjoint_records() -> None:
    assert len(DS1) == 22
    assert len(DS2) == 22
    assert DS1.isdisjoint(DS2)


def test_splits_and_paced_cover_all_48_records() -> None:
    assert DS1 | DS2 | PACED == ALL_MITDB_RECORDS
    assert PACED.isdisjoint(DS1 | DS2)


def test_val_records_are_subset_of_ds1() -> None:
    assert VAL_RECORDS <= DS1
    assert TRAIN_RECORDS == DS1 - VAL_RECORDS


def test_minority_class_records_stay_in_training() -> None:
    # 208 holds ~90% of DS1's F beats and 209 most of DS1's S beats.
    assert {208, 209} <= TRAIN_RECORDS


def test_assert_disjoint_patients_raises_on_overlap() -> None:
    # Arrange
    groups = {"train": [101, 106], "test": [106, 100]}

    # Act / Assert
    with pytest.raises(PatientOverlapError, match="106"):
        assert_disjoint_patients(groups)


def test_assert_disjoint_patients_accepts_real_split() -> None:
    assert_disjoint_patients({"train": TRAIN_RECORDS, "val": VAL_RECORDS, "test": DS2})
