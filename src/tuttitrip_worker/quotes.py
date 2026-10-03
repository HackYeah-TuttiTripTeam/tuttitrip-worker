"""Verbatim quote check shared by every domain that quotes pasted text.

A model may only point at text that is really in the document. :func:`find_quote`
is the one rule: it never accepts a paraphrase. This module is pure (standard
library only), so both a domain's ``logic`` and its steps may import it.
"""

import re


def find_quote(text: str, quote: str) -> str | None:
    """Find a model's quote in the document and return the document's own text.

    The match is exact (case and punctuation count). Only runs of whitespace
    may differ, because models fold line breaks; the returned span is cut from
    ``text``, so it is always an exact substring of it.

    Args:
        text: The pasted document.
        quote: What the model claims is in it.

    Returns:
        The matching span of ``text``, or ``None`` when the quote is not there
        (or is empty).
    """
    words = quote.split()
    if not words:
        return None
    pattern = r"\s+".join(re.escape(word) for word in words)
    found = re.search(pattern, text)
    return found.group(0) if found else None
