"""Checks that a justification states nothing the algorithm did not compute.

Rules for the text of one place (a violation makes the agent retry):

* at most ``MAX_SENTENCES`` sentences and ``MAX_JUSTIFICATION_CHARS`` characters,
* every number is one of the given numbers (a fraction may be written as a
  percent, rounded to at most ``MAX_DECIMALS`` decimals) or a count of the
  given lists,
* every capitalised word inside a sentence belongs to a given name (people,
  the place, its substitute); Polish declension is allowed by comparing the
  word with the name without its last letters,
* the answer has exactly one entry per given place and no other place.
"""

import math
import re
from collections.abc import Iterable, Sequence

from tuttitrip_worker.planning.constants import (
    MAX_DECIMALS,
    MAX_JUSTIFICATION_CHARS,
    MAX_SENTENCES,
    MIN_NAME_STEM,
    NAME_ENDING_CHARS,
    NUMBER_PATTERN,
    NUMBER_TOLERANCE,
    PERCENT,
    SENTENCE_END_PATTERN,
    WORD_PATTERN,
)
from tuttitrip_worker.planning.schemas import JustificationItem, VerdictFacts

_NUMBER = re.compile(NUMBER_PATTERN)
_WORD = re.compile(WORD_PATTERN)
_SENTENCE_END = re.compile(SENTENCE_END_PATTERN)


def _variants(value: float) -> set[float]:
    # A value as given, rounded, and a fraction also as a percent.
    scales = (1.0, float(PERCENT)) if abs(value) <= 1 else (1.0,)
    return {
        round(abs(value) * scale, digits)
        for scale in scales
        for digits in range(MAX_DECIMALS + 1)
    }


def _names(fact: VerdictFacts) -> list[str]:
    names = [p.name for p in (*fact.yes, *fact.no, *fact.people)]
    names += [n for n in (fact.place_name, fact.substitute_name) if n]
    return names


def _parse(number: str) -> float:
    return float(number.replace(",", "."))


def allowed_numbers(fact: VerdictFacts) -> set[float]:
    """Numbers a text about this verdict may contain.

    Args:
        fact: The verdict data given to the model.

    Returns:
        The given numbers in every allowed notation, the list counts and the
        digits that belong to names.
    """
    people = {p.name for p in (*fact.yes, *fact.no, *fact.people)}
    allowed: set[float] = {
        float(len(fact.yes)),
        float(len(fact.no)),
        float(len(people)),
    }
    if fact.v_p is not None:
        allowed |= _variants(fact.v_p)
    for person in fact.people:
        for value in (person.match, person.effort, person.utility):
            allowed |= _variants(value)
    for name in _names(fact):
        allowed |= {_parse(found) for found in _NUMBER.findall(name)}
    return allowed


def _stems(fact: VerdictFacts) -> set[str]:
    stems: set[str] = set()
    for name in _names(fact):
        for word in _WORD.findall(name):
            cut = max(MIN_NAME_STEM, len(word) - NAME_ENDING_CHARS)
            stems.add(word.casefold()[:cut])
    return stems


def _foreign_names(text: str, stems: set[str]) -> list[str]:
    found: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        for word in _WORD.findall(sentence)[1:]:
            known = any(word.casefold().startswith(stem) for stem in stems)
            if word[0].isupper() and not known:
                found.append(word)
    return found


def check_text(text: str, fact: VerdictFacts) -> list[str]:
    """Problems of one justification.

    Args:
        text: What the model wrote.
        fact: The verdict data it was written from.

    Returns:
        Human-readable problems for the model; empty when the text is fine.
    """
    if not text.strip():
        return ["the text is empty"]
    problems: list[str] = []
    if len(text) > MAX_JUSTIFICATION_CHARS:
        problems.append(f"longer than {MAX_JUSTIFICATION_CHARS} characters")
    if len([s for s in _SENTENCE_END.split(text.strip()) if s]) > MAX_SENTENCES:
        problems.append(f"more than {MAX_SENTENCES} sentences")
    allowed = allowed_numbers(fact)
    problems += [
        f"the number {raw} is not in the data"
        for raw in _NUMBER.findall(text)
        if not any(
            math.isclose(_parse(raw), ok, abs_tol=NUMBER_TOLERANCE) for ok in allowed
        )
    ]
    problems += [
        f"the name {word} is not in the data"
        for word in _foreign_names(text, _stems(fact))
    ]
    return problems


def check_batch(
    items: Iterable[JustificationItem], facts: Sequence[VerdictFacts]
) -> list[str]:
    """Problems of a whole model answer for a batch.

    Args:
        items: The model's entries.
        facts: The verdicts of the batch.

    Returns:
        Problems, each naming the place; empty when the answer is fine.
    """
    by_id = {fact.place_id: fact for fact in facts}
    problems: list[str] = []
    seen: set[str] = set()
    for item in items:
        fact = by_id.get(item.place_id)
        if fact is None:
            problems.append(f"{item.place_id}: not a place of this request")
        elif item.place_id in seen:
            problems.append(f"{item.place_id}: written twice")
        else:
            problems += [f"{item.place_id}: {p}" for p in check_text(item.text, fact)]
        seen.add(item.place_id)
    problems += [f"{place_id}: missing" for place_id in by_id.keys() - seen]
    return sorted(problems)


def passing(
    items: Iterable[JustificationItem], facts: Sequence[VerdictFacts]
) -> list[JustificationItem]:
    """The entries that name a given place once and pass every check.

    Args:
        items: The model's entries.
        facts: The verdicts of the batch.

    Returns:
        Entries to keep, in the model's order; the rest has no justification.
    """
    by_id = {fact.place_id: fact for fact in facts}
    kept: dict[str, JustificationItem] = {}
    for item in items:
        fact = by_id.get(item.place_id)
        if fact and item.place_id not in kept and not check_text(item.text, fact):
            kept[item.place_id] = item
    return list(kept.values())
