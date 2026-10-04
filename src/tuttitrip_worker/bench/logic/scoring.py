"""Deterministic scoring of answers against the reference, per case (pure).

Every scorer returns :class:`Scored`: ``parts`` are field checks done in code
(amounts, dates, ids, verdicts), and ``judge`` holds the fuzzy remainder the
LLM judge scores against the rubric. A reference field that is ``None`` means
"the source does not say": the answer must leave it empty too.
"""

from typing import Any

from tuttitrip_worker.bench.constants import (
    ALTERNATIVE_SEPARATOR,
    JUDGE_WEIGHT_EXPENSE,
    JUDGE_WEIGHT_OFFER,
    JUDGE_WEIGHT_PLAN,
    JUDGE_WEIGHT_RECEIPT,
    JUDGE_WEIGHT_TRIP,
    MAX_HIGHLIGHTS,
    MAX_SCORE,
    MIN_SCORE,
    STATE_CONFLICTING,
    STATE_UNCONFIRMED,
    VERDICT_ABSENT,
    VERDICT_NOT_APPLICABLE,
    VERDICT_PRESENT,
)
from tuttitrip_worker.bench.logic.compare import (
    exact,
    get_list,
    mean,
    name_set,
    text_covers,
    text_equal,
)
from tuttitrip_worker.bench.schemas import Example, Json, JudgeView, Scored

PLAN_FIELDS = (
    "day",
    "start_time",
    "end_time",
    "amount_minor",
    "currency",
    "transport",
)
"""Fields of a parsed plan item that code checks one by one."""


def _scored(parts: dict[str, float], judge: JudgeView | None, weight: float) -> Scored:
    return Scored(
        deterministic=mean(list(parts.values())),
        parts=parts,
        judge=judge,
        judge_weight=weight if judge is not None else 0.0,
    )


def requirement_state(verdicts: list[str | None]) -> str:
    """What the quotes of one requirement say together (the backend's reading).

    Args:
        verdicts: Verdict per quote (``None`` = the judge was unavailable).

    Returns:
        ``unconfirmed`` (no quotes), ``present``, ``absent``, ``not_applicable``,
        ``conflicting`` (present and absent), or ``unjudged``.
    """
    if not verdicts:
        return STATE_UNCONFIRMED
    known = {v for v in verdicts if v is not None}
    if not known:
        return "unjudged"
    decisive = known - {VERDICT_NOT_APPLICABLE}
    if {VERDICT_PRESENT, VERDICT_ABSENT} <= decisive:
        return STATE_CONFLICTING
    return next(iter(decisive)) if decisive else VERDICT_NOT_APPLICABLE


def _quotes_of(entry: Any) -> list[dict[str, Any]]:  # ruff: ignore[any-type] - JSON
    quotes = get_list(entry, "quotes") if isinstance(entry, dict) else []
    return [q for q in quotes if isinstance(q, dict)]


def _recall(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> float:
    found = [str(q.get("text", "")) for q in actual]
    hits = [
        any(text_covers(str(want.get("text", "")), text) for text in found)
        for want in expected
    ]
    return mean([MAX_SCORE if hit else MIN_SCORE for hit in hits])


def score_offer(example: Example, answer: Json) -> Scored:
    """Score the evidence of a pasted offer.

    Per requirement: the state the quotes add up to, and whether every
    reference quote is covered by an answer quote. The judge scores whether
    the quotes are relevant to the requirement.

    Args:
        example: Golden example; ``expected["evidence"]`` maps key to quotes.
        answer: ``{"evidence": {key: {"quotes": [...]}}}``.

    Returns:
        The scored answer.
    """
    wanted: dict[str, Any] = example.expected["evidence"]
    got_raw = answer.get("evidence")
    got: dict[str, Any] = got_raw if isinstance(got_raw, dict) else {}
    parts: dict[str, float] = {}
    for key, entry in wanted.items():
        want_quotes, got_quotes = _quotes_of(entry), _quotes_of(got.get(key))
        want_state = requirement_state([q.get("verdict") for q in want_quotes])
        got_state = requirement_state([q.get("verdict") for q in got_quotes])
        parts[f"{key}:state"] = exact(want_state, got_state)
        if want_quotes:
            parts[f"{key}:quotes"] = _recall(want_quotes, got_quotes)
    view = JudgeView(
        criteria=(
            "For each requirement: do the candidate's quotes concern the "
            "requirement and carry the same information as the reference quotes? "
            "A requirement the reference leaves without quotes must have none. "
            "Quotes about something else, or extra irrelevant ones, lower the score."
        ),
        reference={
            key: [q.get("text") for q in _quotes_of(entry)]
            for key, entry in wanted.items()
        },
        candidate={
            key: [q.get("text") for q in _quotes_of(entry)]
            for key, entry in got.items()
        },
    )
    return _scored(parts, view, JUDGE_WEIGHT_OFFER)


def score_verdict(example: Example, answer: Json) -> Scored:
    """Score the judge stage: one verdict per quote, compared one by one.

    Args:
        example: Golden example; ``expected["verdicts"]`` lists the verdicts.
        answer: ``{"verdicts": [...]}``.

    Returns:
        The scored answer; there is nothing for the LLM judge to add.
    """
    want = example.expected["verdicts"]
    got = get_list(answer, "verdicts")
    parts = {
        f"quote_{number}": exact(verdict, got[number] if number < len(got) else None)
        for number, verdict in enumerate(want)
    }
    return _scored(parts, None, 0.0)


def _align(wanted: list[Json], actual: list[Json]) -> list[tuple[Json, Json | None]]:
    """Pair each reference item with the answer item that names the same place.

    Args:
        wanted: Reference items, in text order.
        actual: Answer items.

    Returns:
        ``(reference, answer or None)`` per reference item; an answer item is
        used at most once.
    """
    free = list(actual)
    pairs: list[tuple[Json, Json | None]] = []
    for want in wanted:
        match = next(
            (
                item
                for item in free
                if text_covers(want.get("place_name"), item.get("place_name"))
            ),
            None,
        )
        if match is not None:
            free.remove(match)
        pairs.append((want, match))
    return pairs


def score_plan(example: Example, answer: Json) -> Scored:
    """Score a parsed pasted plan.

    Items are paired by place name. Parts: recall and precision of the items,
    and the fields of paired items (day, times, price, currency, transport).
    The judge scores the addresses and cities.

    Args:
        example: Golden example; ``expected["items"]`` are the reference items.
        answer: ``{"items": [...]}`` as the worker returns it.

    Returns:
        The scored answer.
    """
    wanted: list[Json] = example.expected["items"]
    actual = [item for item in get_list(answer, "items") if isinstance(item, dict)]
    pairs = _align(wanted, actual)
    paired = [(want, got) for want, got in pairs if got is not None]
    parts = {
        "recall": len(paired) / len(wanted) if wanted else MAX_SCORE,
        "precision": len(paired) / len(actual) if actual else MAX_SCORE,
        "fields": mean(
            [
                exact(want.get(field), got.get(field) if got else None)
                if got
                else MIN_SCORE
                for want, got in pairs
                for field in PLAN_FIELDS
            ]
        ),
    }
    view = JudgeView(
        criteria=(
            "Compare address and city of each place. The candidate may write "
            "them shorter or longer (street without 'ul.', city omitted when the "
            "reference city equals the plan's city) but must not change or invent "
            "them. A missing item lowers the score."
        ),
        reference={
            "items": [
                {k: w.get(k) for k in ("place_name", "address", "city")} for w in wanted
            ]
        },
        candidate={
            "items": [
                {k: g.get(k) for k in ("place_name", "address", "city")} for g in actual
            ]
        },
    )
    return _scored(parts, view if wanted else None, JUDGE_WEIGHT_PLAN)


def score_trip(example: Example, answer: Json) -> Scored:
    """Score a generated trip plan.

    Code checks the destination, the number of days (when the request states
    it) and the size of the highlights list; the judge scores whether the
    highlights fit the request.

    Args:
        example: Golden example; ``expected`` has ``destination`` (acceptable
            names joined with ``|``, or ``None`` when any place is fine),
            ``days`` (or ``None``), ``highlights`` (reference ideas) and the
            judge's ``criteria``.
        answer: ``{"destination", "days", "highlights"}``.

    Returns:
        The scored answer.
    """
    highlights = [str(h) for h in get_list(answer, "highlights")]
    parts = {
        "highlights": MAX_SCORE if 0 < len(highlights) <= MAX_HIGHLIGHTS else MIN_SCORE,
    }
    destination = example.expected["destination"]
    if destination is not None:
        alternatives = destination.split(ALTERNATIVE_SEPARATOR)
        parts["destination"] = max(
            text_covers(option, str(answer.get("destination", "")))
            for option in alternatives
        )
    if example.expected["days"] is not None:
        parts["days"] = exact(example.expected["days"], answer.get("days"))
    view = JudgeView(
        criteria=str(example.expected["criteria"]),
        reference={"highlights": example.expected["highlights"]},
        candidate={"highlights": highlights, "destination": answer.get("destination")},
    )
    return _scored(parts, view, JUDGE_WEIGHT_TRIP)


def score_expense(example: Example, answer: Json) -> Scored:
    """Score a parsed expense sentence.

    Args:
        example: Golden example; ``expected`` has the amount in minor units,
            currency, payer, included and excluded names and a description.
        answer: The reading as the worker returns it.

    Returns:
        The scored answer; the judge scores the description.
    """
    want = example.expected
    parts = {
        "amount": exact(want["amount_minor"], answer.get("amount_minor")),
        "currency": exact(want["currency"], answer.get("currency")),
        "payer": text_equal(want["payer_name"], answer.get("payer_name")),
        "included": name_set(
            want["included_names"], map(str, get_list(answer, "included_names"))
        ),
        "excluded": name_set(
            want["excluded_names"], map(str, get_list(answer, "excluded_names"))
        ),
    }
    view = JudgeView(
        criteria=(
            "Is the candidate description a faithful short note of what was "
            "bought, as in the sentence? It may differ in wording but must not "
            "add facts."
        ),
        reference={"description": want["description"]},
        candidate={"description": answer.get("description")},
    )
    return _scored(parts, view, JUDGE_WEIGHT_EXPENSE)


def score_receipt(example: Example, answer: Json) -> Scored:
    """Score a receipt reading.

    Args:
        example: Golden example; ``expected`` has the total in minor units,
            currency, date, merchant and the printed lines.
        answer: The reading as the worker returns it.

    Returns:
        The scored answer; the judge scores the item names.
    """
    want = example.expected
    got_items = [i for i in get_list(answer, "items") if isinstance(i, dict)]
    want_totals = sorted(i["amount_minor"] for i in want["items"])
    got_totals = sorted(i.get("amount_minor") for i in got_items)
    parts = {
        "amount": exact(want["amount_minor"], answer.get("amount_minor")),
        "currency": exact(want["currency"], answer.get("currency")),
        "date": exact(want["spent_on"], answer.get("spent_on")),
        "merchant": text_covers(want["merchant"], answer.get("merchant")),
        "line_totals": exact(want_totals, got_totals),
    }
    view = JudgeView(
        criteria=(
            "Do the candidate item names read as the printed lines of the "
            "reference (spelling of OCR noise and abbreviations is fine)? "
            "Invented or missing lines lower the score."
        ),
        reference={"items": [i["name"] for i in want["items"]]},
        candidate={"items": [i.get("name") for i in got_items]},
    )
    return _scored(parts, view if want["items"] else None, JUDGE_WEIGHT_RECEIPT)


def score_match(example: Example, answer: Json) -> Scored:
    """Score a place match: the chosen catalog id (or no place) must be right.

    Args:
        example: Golden example; ``expected["place_id"]`` is an id or ``None``.
        answer: ``{"place_id": id or None}``.

    Returns:
        The scored answer; the id is checked in code.
    """
    parts = {"place_id": exact(example.expected["place_id"], answer.get("place_id"))}
    return _scored(parts, None, 0.0)


def combine(scored: Scored, judge_score: float | None) -> float:
    """Final score of an example.

    Args:
        scored: Deterministic result.
        judge_score: The judge's score, ``None`` when there was no judge.

    Returns:
        ``deterministic`` alone, or the judge-weighted mix.
    """
    if judge_score is None:
        return scored.deterministic
    weight = scored.judge_weight
    return scored.deterministic * (1 - weight) + judge_score * weight
