"""Catalog candidates for a pasted item and the no-model decision (pure).

Pure code ranks the places of the item's city by the similarity of their names
(``difflib`` from the standard library, after folding case and diacritics and
dropping generic words such as "muzeum"); a decision model only picks one of
the few best. The order is deterministic: score, then place id.
"""

import re
import unicodedata
from collections.abc import Iterable, Sequence
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Final

from tuttitrip_worker.contracts import MatchCandidate, PlaceMatch
from tuttitrip_worker.linter.schemas import CatalogPlace, RankedPlace

MAX_CANDIDATES: Final = 9
"""The decision model takes 10 options; the 10th is "none of these"."""

MIN_CANDIDATE_SCORE: Final = 0.45
"""Below this similarity a place is not even offered as a candidate."""

AUTO_MATCH_SCORE: Final = 0.85
"""Without a model, the best candidate at or above this is a match."""

MODEL_CONFIDENCE_FLOOR: Final = 0.5
"""Below this margin (0..1) a model pick must be confirmed by the host."""

GENERIC_WORDS: Final = frozenset(
    {
        # Polish, English, German words that name a kind of place, not the place.
        *("muzeum", "museum", "park", "kosciol", "church", "kirche", "zamek"),
        *("castle", "schloss", "rynek", "plac", "platz", "square", "ulica"),
        *("restauracja", "restaurant", "cafe", "kawiarnia", "hotel", "galeria"),
        *("gallery", "galerie", "the", "der", "die", "das", "w", "we", "na", "im"),
    }
)

_EXTRA_FOLDS: Final = str.maketrans({"ł": "l", "ß": "ss", "ø": "o", "đ": "d"})
_WORD = re.compile(r"[^\W_]+")


@lru_cache(maxsize=65536)
def fold(text: str) -> str:
    """Lower-case, strip diacritics (``ł`` becomes ``l``) and collapse spaces.

    Args:
        text: A place name.

    Returns:
        Words of ASCII-ish letters and digits joined by single spaces.
    """
    lowered = text.casefold().translate(_EXTRA_FOLDS)
    plain = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(c for c in plain if not unicodedata.combining(c))
    return " ".join(match.group() for match in _WORD.finditer(stripped))


@lru_cache(maxsize=65536)
def core(folded: str) -> str:
    """Drop generic words; keep everything when nothing else is left.

    Args:
        folded: Output of :func:`fold`.

    Returns:
        The distinctive part of the name.
    """
    words = [w for w in folded.split() if w not in GENERIC_WORDS]
    return " ".join(words) or folded


def _ratio(a: str, b: str) -> float:
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    # Both are upper bounds of ratio(): cheap rejection of the many unlike names.
    if (
        matcher.real_quick_ratio() < MIN_CANDIDATE_SCORE
        or matcher.quick_ratio() < MIN_CANDIDATE_SCORE
    ):
        return 0.0
    return matcher.ratio()


def similarity(item_name: str, place_name: str) -> float:
    """How alike two names are, 0..1 (the better of full and distinctive parts).

    Args:
        item_name: Name from the pasted plan.
        place_name: Name in the catalog.

    Returns:
        Similarity; 1.0 for names equal after folding.
    """
    a, b = fold(item_name), fold(place_name)
    if not a or not b:
        return 0.0
    return max(_ratio(a, b), _ratio(core(a), core(b)))


def rank(
    item_name: str,
    places: Iterable[CatalogPlace],
    limit: int = MAX_CANDIDATES,
) -> list[RankedPlace]:
    """Best catalog places for one item, deterministic.

    Args:
        item_name: Name from the pasted plan.
        places: Places of the item's city.
        limit: How many to return at most.

    Returns:
        Up to ``limit`` places scoring at least ``MIN_CANDIDATE_SCORE``, best
        first; ties are broken by place id.
    """
    scored = [
        RankedPlace(
            place_id=place.place_id,
            name=place.name,
            category=place.category,
            score=round(similarity(item_name, place.name), 4),
        )
        for place in places
    ]
    good = [p for p in scored if p.score >= MIN_CANDIDATE_SCORE]
    good.sort(key=lambda p: (-p.score, p.place_id))
    return good[:limit]


def to_candidates(ranked: Sequence[RankedPlace]) -> list[MatchCandidate]:
    """Contract view of ranked places.

    Args:
        ranked: Output of :func:`rank`.

    Returns:
        Candidates for ``PlaceMatch.candidates``.
    """
    return [
        MatchCandidate(
            place_id=p.place_id, name=p.name[:200], category=p.category, score=p.score
        )
        for p in ranked
    ]


def match_without_model(item_index: int, ranked: Sequence[RankedPlace]) -> PlaceMatch:
    """Decide by the similarity threshold alone.

    Args:
        item_index: Index of the pasted item.
        ranked: Candidates of the item, best first.

    Returns:
        ``matched`` when the best one is at or above ``AUTO_MATCH_SCORE``,
        otherwise ``unrecognized``; the candidates are kept either way.
    """
    candidates = to_candidates(ranked)
    best = ranked[0] if ranked else None
    if best is not None and best.score >= AUTO_MATCH_SCORE:
        return PlaceMatch(
            item_index=item_index,
            status="matched",
            place_id=best.place_id,
            confidence=best.score,
            candidates=candidates,
        )
    return PlaceMatch(
        item_index=item_index, status="unrecognized", candidates=candidates
    )


def match_with_pick(
    item_index: int,
    ranked: Sequence[RankedPlace],
    picked: RankedPlace | None,
    confidence: float | None,
) -> PlaceMatch:
    """Turn the decision model's pick into a match.

    A pick without a reported confidence (a language-model fallback answered)
    counts as low confidence. "None of these" is ``unrecognized``.

    Args:
        item_index: Index of the pasted item.
        ranked: Candidates the model was asked about.
        picked: The candidate it chose, or ``None`` for "none of these".
        confidence: Its margin 0..1, if it reported one.

    Returns:
        ``matched``, ``needs_confirmation`` (the pick is kept in ``place_id``)
        or ``unrecognized``.
    """
    candidates = to_candidates(ranked)
    if picked is None:
        return PlaceMatch(
            item_index=item_index,
            status="unrecognized",
            confidence=confidence,
            candidates=candidates,
        )
    sure = confidence is not None and confidence >= MODEL_CONFIDENCE_FLOOR
    return PlaceMatch(
        item_index=item_index,
        status="matched" if sure else "needs_confirmation",
        place_id=picked.place_id,
        confidence=confidence,
        candidates=candidates,
    )
