"""Money amounts as written by people (pure).

The model may only report an amount the text really contains, so the worker
parses the written form itself and compares it with the model's number.
"""

import re
from decimal import Decimal, InvalidOperation

_NOISE = re.compile(r"[^\d.,]")
_MINOR_UNITS = 100
_DECIMAL_DIGITS = 2
_GROUP_DIGITS = 3


def parse_amount(written: str) -> int | None:
    """Read an amount such as ``1 250,50 zł`` or ``12.5 EUR`` as minor units.

    A separator followed by one or two digits is the decimal point; followed by
    three digits it groups thousands (``1.250`` is 1250). When both ``,`` and
    ``.`` occur, the last one is the decimal point.

    Args:
        written: The amount as written, with or without a currency.

    Returns:
        Minor units (grosze, cents), or ``None`` when no single amount is there.
    """
    cleaned = _NOISE.sub("", written)
    if not cleaned or not re.fullmatch(r"\d[\d.,]*", cleaned):
        return None
    last = max(cleaned.rfind(","), cleaned.rfind("."))
    tail = cleaned[last + 1 :]
    both = "," in cleaned and "." in cleaned
    if last == -1 or (len(tail) == _GROUP_DIGITS and not both):
        whole, fraction = cleaned, ""
    else:
        whole, fraction = cleaned[:last], tail
    whole = re.sub(r"[.,]", "", whole)
    if len(fraction) > _DECIMAL_DIGITS or not whole:
        return None
    try:
        value = Decimal(f"{whole}.{fraction or '0'}")
    except InvalidOperation:
        return None
    return int(value * _MINOR_UNITS)


def amount_is_in_text(text: str, written: str, amount_minor: int) -> bool:
    """Whether the amount the model reported is written in the text.

    Args:
        text: The user's sentence.
        written: The fragment of the text the model says holds the amount.
        amount_minor: The number the model reported.

    Returns:
        ``True`` when the fragment occurs in the text and parses to the number.
    """
    fragment = written.strip()
    return (
        bool(fragment)
        and fragment.casefold() in text.casefold()
        and parse_amount(fragment) == amount_minor
    )
