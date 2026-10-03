"""City slug by the rule of ``contracts.SLUG_PATTERN`` (pure)."""

import re
import unicodedata

# Letters NFKD does not decompose (they have no combining mark).
_TRANSLITERATION = str.maketrans(
    {"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss"}
)
_NON_WORD = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Build a city slug from free text.

    Lowercases, removes diacritics (``ł`` becomes ``l``), turns every run of
    other characters into one ``-`` and trims ``-`` at both ends.

    Args:
        text: For example ``"Gdańsk, Polska"``.

    Returns:
        For example ``"gdansk-polska"``; empty when nothing usable is left.
    """
    plain = unicodedata.normalize("NFKD", text.translate(_TRANSLITERATION))
    ascii_only = plain.encode("ascii", "ignore").decode("ascii").lower()
    return _NON_WORD.sub("-", ascii_only).strip("-")
