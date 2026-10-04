"""Comparison helpers: text folding and field-by-field scoring (pure)."""

import re
import unicodedata
from collections.abc import Collection, Iterable
from typing import Any, Final

from tuttitrip_worker.bench.constants import MAX_SCORE, MIN_SCORE

_EXTRA_FOLDS: Final = str.maketrans({"ł": "l", "ß": "ss", "ø": "o", "đ": "d"})
_WORD: Final = re.compile(r"[^\W_]+")


def fold(text: str | None) -> str:
    """Lower-case, strip diacritics and collapse spaces.

    Args:
        text: Any text, ``None`` counts as empty.

    Returns:
        Words of plain letters and digits joined by single spaces.
    """
    lowered = (text or "").casefold().translate(_EXTRA_FOLDS)
    plain = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(char for char in plain if not unicodedata.combining(char))
    return " ".join(match.group() for match in _WORD.finditer(stripped))


def exact(expected: object, actual: object) -> float:
    """1 when both are equal (``None`` equals ``None``), else 0.

    Args:
        expected: Reference value.
        actual: Answer value.

    Returns:
        1.0 or 0.0.
    """
    return MAX_SCORE if expected == actual else MIN_SCORE


def text_equal(expected: str | None, actual: str | None) -> float:
    """1 when both are equal after folding (both empty counts as equal).

    Args:
        expected: Reference text.
        actual: Answer text.

    Returns:
        1.0 or 0.0.
    """
    return exact(fold(expected), fold(actual))


def text_covers(expected: str | None, actual: str | None) -> float:
    """1 when the folded texts are equal or one contains the other.

    Args:
        expected: Reference text.
        actual: Answer text.

    Returns:
        1.0 or 0.0; two empty texts are equal.
    """
    left, right = fold(expected), fold(actual)
    if not left or not right:
        return exact(left, right)
    return MAX_SCORE if left in right or right in left else MIN_SCORE


def name_set(expected: Iterable[str], actual: Iterable[str]) -> float:
    """F1 of two sets of names, compared after folding.

    Args:
        expected: Reference names.
        actual: Answer names.

    Returns:
        1.0 when both are empty or equal, else the F1 score.
    """
    want = {fold(name) for name in expected} - {""}
    got = {fold(name) for name in actual} - {""}
    if not want and not got:
        return MAX_SCORE
    hit = len(want & got)
    if hit == 0:
        return MIN_SCORE
    precision, recall = hit / len(got), hit / len(want)
    return 2 * precision * recall / (precision + recall)


def mean(values: Collection[float]) -> float:
    """Arithmetic mean; an empty collection scores 1 (nothing to get wrong).

    Args:
        values: Scores.

    Returns:
        The mean.
    """
    return sum(values) / len(values) if values else MAX_SCORE


def clamp(value: float) -> float:
    """Keep a score within 0..1.

    Args:
        value: Any number.

    Returns:
        The number limited to the score range.
    """
    return max(MIN_SCORE, min(MAX_SCORE, value))


def get_list(source: dict[str, Any], key: str) -> list[Any]:
    """Read a list field of an untrusted answer; anything else is empty.

    Args:
        source: A JSON object.
        key: Field name.

    Returns:
        The list, or an empty one.
    """
    value = source.get(key)
    return value if isinstance(value, list) else []
