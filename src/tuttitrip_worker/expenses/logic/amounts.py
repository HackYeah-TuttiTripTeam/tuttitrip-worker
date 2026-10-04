"""Money amounts as written by people (pure).

The model may only report an amount the text really contains, so the worker
finds the numbers in the text itself, parses them and requires the model's
amount to be one of them. Numbers are matched as whole tokens: ``120`` is not
inside ``120,50``.
"""

import re
from decimal import Decimal
from typing import Final

_MINOR_UNITS = 100
_SPACES = r"[ \u00a0]"
_NUMBER = (
    r"-?(?:"
    rf"\d{{1,3}}(?:{_SPACES}\d{{3}})+(?:[.,]\d{{1,2}})?"  # 1 250,50
    r"|\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?"  # 1.250,50
    r"|\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?"  # 1,250.50
    r"|\d+(?:[.,]\d{1,2})?"  # 1250,50
    r")"
)
_WHOLE: Final = re.compile(_NUMBER)
# A token starts after a space, a start, or punctuation that is not part of a
# number or a word, and ends before a digit or a decimal part that did not fit.
_TOKEN: Final = re.compile(rf"(?<![\w.,])(?:{_NUMBER})(?![\d]|[.,]\d)")
_ENDS: Final = re.compile(r"^[^\d-]+|[^\d]+$")


def parse_amount(written: str) -> int | None:
    """Read an amount such as ``1 250,50 zł`` or ``12.5 EUR`` as minor units.

    One anchored pattern decides: the fragment must be a single number with an
    optional currency before or after it. A space (or no-break space) may only
    sit between a digit and an exact group of three digits; ``.`` or ``,``
    followed by one or two digits is the decimal point, followed by three digits
    a thousands separator (``1.250`` is 1250). A leading minus is kept.

    Args:
        written: The amount as written, with or without a currency.

    Returns:
        Minor units (grosze, cents), negative for a minus sign, or ``None``
        when the fragment is not exactly one number (``20-30 zł``).
    """
    core = _ENDS.sub("", written.strip()).strip()
    if _WHOLE.fullmatch(core) is None:
        return None
    return _to_minor(core)


def _to_minor(token: str) -> int:
    digits = re.sub(_SPACES, "", token)
    decimal = re.search(r"[.,]\d{1,2}$", digits)
    if decimal and not re.search(r"[.,]\d{3}$", digits):
        whole, fraction = digits[: decimal.start()], decimal.group()[1:]
    else:
        whole, fraction = digits, ""
    whole = re.sub(r"[.,]", "", whole)
    return int(Decimal(f"{whole}.{fraction or '0'}") * _MINOR_UNITS)


def amounts_in(text: str) -> set[int]:
    """Every number written in the text, as minor units.

    Args:
        text: The user's sentence.

    Returns:
        The distinct values of the whole number tokens (dates and counts
        included: the caller picks the one the model named).
    """
    return {_to_minor(token) for token in _tokens(text)}


def _tokens(text: str) -> list[str]:
    return [match.group() for match in _TOKEN.finditer(text)]


def amount_is_in_text(text: str, written: str, amount_minor: int) -> bool:
    """Whether the amount the model reported is written in the text.

    The fragment the model quotes must parse to the reported number, and its
    number must be one of the whole number tokens of the text, character for
    character (so neither ``120`` nor ``120,5`` is accepted for ``120,50 zł``).

    Args:
        text: The user's sentence.
        written: The fragment of the text the model says holds the amount.
        amount_minor: The number the model reported.

    Returns:
        ``True`` when both hold.
    """
    fragment = written.strip()
    number = _ENDS.sub("", fragment).strip()
    return (
        bool(fragment)
        and fragment.casefold() in text.casefold()
        and parse_amount(fragment) == amount_minor
        and number in _tokens(text)
    )


def require_amount(text: str, amount_minor: int | None, written: str) -> int | None:
    """The model's amount if the text really holds it, else ``None``.

    The one check used by the output validator and by the workflow.

    Args:
        text: The user's sentence.
        amount_minor: The number the model reported (may be missing).
        written: The fragment of the text the model says holds the amount.

    Returns:
        A positive amount in minor units, or ``None``.
    """
    if amount_minor is None or amount_minor <= 0:
        return None
    if len(priced_amounts(text)) > 1:
        return None  # several prices in one sentence: never pick one by guessing
    return amount_minor if amount_is_in_text(text, written, amount_minor) else None


CURRENCY_WORDS: Final = {
    "PLN": ("pln", "zł", "zl", "zlotych", "złotych", "złote", "zloty", "złoty"),
    "EUR": ("eur", "€", "euro"),
    "USD": ("usd", "$", "dolar", "dolarów", "dollar", "dollars"),
    "GBP": ("gbp", "£", "funt", "funtów", "pound", "pounds"),
    "CHF": ("chf", "frank", "franków", "franc", "francs"),
    "CZK": ("czk", "kč", "koron", "koruna", "korun"),
    "HUF": ("huf", "ft", "forint", "forintów"),
}
"""How people write currencies (lower case). A reported code is kept only when
the text names it by one of these or by its ISO code."""


def priced_amounts(text: str) -> set[int]:
    """Amounts written with a currency next to them (``120 zł``, ``€5``).

    Args:
        text: The user's sentence.

    Returns:
        The distinct values; more than one makes the sentence ambiguous.
    """
    words = sorted(
        {w for names in CURRENCY_WORDS.values() for w in names}
        | {c.casefold() for c in CURRENCY_WORDS},
        key=len,
        reverse=True,
    )
    alternatives = "|".join(re.escape(word) for word in words)
    after = rf"(?<![\w.,])({_NUMBER})\s*(?:{alternatives})(?!\w)"
    before = rf"(?<!\w)(?:{alternatives})\s*({_NUMBER})(?![\d]|[.,]\d)"
    lowered = text.casefold()
    found = re.findall(after, lowered) + re.findall(before, lowered)
    return {_to_minor(token) for token in found}


def currency_in_text(text: str, code: str | None) -> str | None:
    """Keep a currency only if the text names it.

    Args:
        text: The user's sentence.
        code: ISO 4217 code reported by the model.

    Returns:
        ``code`` when the code, its symbol or a name of it occurs in the text
        as a whole word; otherwise ``None`` (the trip currency applies).
    """
    if not code:
        return None
    upper = code.upper()
    words = {upper.casefold(), *CURRENCY_WORDS.get(upper, ())}
    lowered = text.casefold()
    for word in words:
        if re.search(rf"(?<!\w){re.escape(word)}(?!\w)", lowered):
            return upper
        if not word.isalpha() and word in lowered:  # symbols: €, $, £
            return upper
    return None


def names_in_text(text: str, names: list[str]) -> list[str]:
    """Keep the names that occur in the text (case-insensitive substring).

    Args:
        text: The user's sentence.
        names: Names the model reported.

    Returns:
        The names as the model wrote them, in order, without repeats.
    """
    lowered = text.casefold()
    kept = (name.strip() for name in names)
    return [n for n in dict.fromkeys(kept) if n and n.casefold() in lowered]
