"""Turn raw extractor output into verified evidence (pure, no model, no I/O).

The model only proposes quotes. A quote exists for the result only if it is a
verbatim span of the offer; a requirement without such a quote gets no quotes
and no assessment (the backend then calls it unconfirmed).
"""

from collections.abc import Mapping, Sequence
from typing import Final

from tuttitrip_worker.accommodation.schemas import ExtractedQuotes
from tuttitrip_worker.contracts import (
    EvidenceQuote,
    OfferVerdict,
    RequirementEvidence,
)
from tuttitrip_worker.quotes import find_quote

MAX_QUOTES_PER_KEY: Final = 3
"""Quotes kept per requirement (each costs one judge call)."""

MAX_QUOTE_CHARS: Final = 1000
"""Longest quote accepted; the contract caps ``EvidenceQuote.text`` the same."""

type Assessment = tuple[OfferVerdict | None, float | None]
"""Verdict of the judge and its confidence (``None`` when it reported none)."""


def verified_quotes(
    offer: str, extracted: ExtractedQuotes, keys: Sequence[str]
) -> dict[str, list[str]]:
    """Keep only quotes that are verbatim in the offer, per requested key.

    Args:
        offer: The pasted offer text.
        extracted: Raw extractor output (may name unknown keys or invent text).
        keys: Requested requirement keys, in output order.

    Returns:
        For every key the exact offer spans, without duplicates, at most
        :data:`MAX_QUOTES_PER_KEY`; an empty list when nothing checks out.
    """
    wanted = set(keys)
    found: dict[str, list[str]] = {key: [] for key in keys}
    for entry in extracted.requirements:
        if entry.requirement_key not in wanted:
            continue
        spans = found[entry.requirement_key]
        for raw in entry.quotes:
            span = find_quote(offer, raw)
            if span is None or len(span) > MAX_QUOTE_CHARS or span in spans:
                continue
            if len(spans) < MAX_QUOTES_PER_KEY:
                spans.append(span)
    return found


def build_evidence(
    keys: Sequence[str],
    quotes: Mapping[str, Sequence[str]],
    assessments: Mapping[tuple[str, str], Assessment],
) -> list[RequirementEvidence]:
    """Assemble the contract output, one entry per requested key.

    Args:
        keys: Requested requirement keys, in output order.
        quotes: Verified quotes per key (:func:`verified_quotes`).
        assessments: Judge result per ``(key, quote)``; a missing pair means
            no assessment (judge unavailable).

    Returns:
        ``RequirementEvidence`` for every key; quotes without an assessment
        have ``verdict`` and ``confidence`` unset.
    """
    evidence: list[RequirementEvidence] = []
    for key in keys:
        items: list[EvidenceQuote] = []
        for text in quotes.get(key, ()):
            verdict, confidence = assessments.get((key, text), (None, None))
            items.append(
                EvidenceQuote(text=text, verdict=verdict, confidence=confidence)
            )
        evidence.append(RequirementEvidence(requirement_key=key, quotes=items))
    return evidence
