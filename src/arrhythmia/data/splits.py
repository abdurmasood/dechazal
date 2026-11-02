"""Inter-patient record split of the MIT-BIH Arrhythmia Database.

DS1/DS2 follow de Chazal, O'Dwyer & Reilly (2004), IEEE TBME 51(7):1196-1206. The four
paced records are excluded, as in that paper and in ANSI/AAMI EC57 practice.
"""

from collections.abc import Iterable, Mapping

DS1: frozenset[int] = frozenset(
    {101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122}
    | {124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230}
)
DS2: frozenset[int] = frozenset(
    {100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210}
    | {212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234}
)
PACED: frozenset[int] = frozenset({102, 104, 107, 217})

# Validation patients are drawn from DS1 only, so DS2 is never seen before final evaluation.
# 208 (most DS1 F beats) and 209 (most DS1 S beats) are deliberately kept in training.
VAL_RECORDS: frozenset[int] = frozenset({114, 124, 201, 205, 215})
TRAIN_RECORDS: frozenset[int] = DS1 - VAL_RECORDS
TEST_RECORDS: frozenset[int] = DS2


class PatientOverlapError(ValueError):
    """Raised when the same patient (record) appears in more than one data split."""


def assert_disjoint_patients(groups: Mapping[str, Iterable[int]]) -> None:
    """Fail fast if any record ID appears in more than one named group.

    Args:
        groups: Split name mapped to the record IDs it contains.

    Raises:
        PatientOverlapError: If two groups share a record ID.
    """
    seen: dict[int, str] = {}
    for name, records in groups.items():
        for record_id in set(int(r) for r in records):
            if record_id in seen:
                raise PatientOverlapError(
                    f"Record {record_id} appears in both '{seen[record_id]}' and '{name}'"
                )
            seen[record_id] = name
