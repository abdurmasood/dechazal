"""Mapping from MIT-BIH annotation symbols to ANSI/AAMI EC57 beat classes."""

from enum import IntEnum


class AamiClass(IntEnum):
    """AAMI beat classes. N, S, V, F have contiguous indices so they double as model labels."""

    N = 0  # normal and bundle branch block beats
    S = 1  # supraventricular ectopic
    V = 2  # ventricular ectopic
    F = 3  # fusion of ventricular and normal
    Q = 4  # paced, fusion of paced and normal, unclassifiable


# Q is too rare outside the paced records (8 beats in DS1, 7 in DS2) to learn, so it is
# counted and excluded from training and evaluation.
TRAINING_CLASSES: tuple[AamiClass, ...] = (AamiClass.N, AamiClass.S, AamiClass.V, AamiClass.F)

AAMI_MAP: dict[str, AamiClass] = {
    **dict.fromkeys(["N", "L", "R", "e", "j"], AamiClass.N),
    **dict.fromkeys(["A", "a", "J", "S"], AamiClass.S),
    **dict.fromkeys(["V", "E"], AamiClass.V),
    "F": AamiClass.F,
    **dict.fromkeys(["/", "f", "Q"], AamiClass.Q),
}

# WFDB annotation codes that mark events, rhythm changes or noise rather than beats.
NON_BEAT_SYMBOLS: frozenset[str] = frozenset(
    ["[", "!", "]", "x", "(", ")", "p", "t", "u", "`", "'", "^", "|", "~", "+"]
    + ["s", "T", "*", "D", "=", '"', "@"]
)


class UnknownSymbolError(ValueError):
    """Raised for an annotation symbol that is neither a known beat nor a known non-beat."""


def to_aami(symbol: str) -> AamiClass | None:
    """Map one MIT-BIH annotation symbol to its AAMI class.

    Args:
        symbol: WFDB annotation symbol, e.g. "N", "V", "+".

    Returns:
        The AAMI class for beat annotations, or None for non-beat annotations.

    Raises:
        UnknownSymbolError: If the symbol is not in either table.
    """
    if symbol in AAMI_MAP:
        return AAMI_MAP[symbol]
    if symbol in NON_BEAT_SYMBOLS:
        return None
    raise UnknownSymbolError(f"Unknown MIT-BIH annotation symbol {symbol!r}")
