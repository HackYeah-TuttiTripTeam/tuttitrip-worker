"""Item checks: a parsed item is only trusted if its text backs it up.

The model may only read. Each item carries a verbatim quote and the checks
here decide, without a model, whether the item is real:

* the quote occurs in the pasted text (:func:`tuttitrip_worker.quotes.find_quote`),
  is one line and at most :data:`QUOTE_MAX_CHARS` long;
* the place name occurs in the quote;
* a start or end time and an amount, when set, are written in the quote.

An item that fails goes to ``unread`` (``quote_not_in_text`` when the quote is
absent, ``invalid_item`` for the rest) instead of the result. Pure: no model,
no I/O.
"""

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from operator import itemgetter
from typing import Final, Literal, NamedTuple

from pydantic import ValidationError

from tuttitrip_worker.contracts import ParsedPlanItem, UnreadItem
from tuttitrip_worker.linter.schemas import DraftPlanItem
from tuttitrip_worker.quotes import find_quote

QUOTE_MAX_CHARS: Final = 300
"""Longest accepted quote: one line or sentence, not a paragraph."""

UNREAD_QUOTE_MAX_CHARS: Final = 1000
"""Longest quote the contract keeps in ``UnreadItem``."""

_TIME = re.compile(r"(\d{1,2}):(\d{2})")
_CLOCK = re.compile(r"\d{1,2}:\d{2}")
_NUMBER = re.compile(r"\d+(?:[ \N{NO-BREAK SPACE}]\d{3})*(?:[.,]\d{1,2})?")


class ItemCheck(NamedTuple):
    """Outcome of :func:`check_item`.

    ``problem`` is ``None`` for a good item; otherwise ``reason`` is the
    ``UnreadItem`` reason and ``problem`` tells the model what to fix.
    ``span`` is the quote cut from the text (``None`` when it was not found).
    """

    span: str | None
    reason: Literal["quote_not_in_text", "invalid_item"] | None
    problem: str | None


def _fold(value: str) -> str:
    return " ".join(value.casefold().split())


def _time_in(quote: str, time: str) -> bool:
    parsed = _TIME.fullmatch(time.strip())
    if parsed is None:
        return False
    hour, minute = int(parsed[1]), parsed[2]
    return re.search(rf"(?<!\d)0?{hour}[:.]{minute}(?!\d)", quote) is not None


def _amount_in(quote: str, amount_minor: int) -> bool:
    for number in _NUMBER.findall(_CLOCK.sub(" ", quote)):
        cleaned = re.sub(r"[ \N{NO-BREAK SPACE}]", "", number).replace(",", ".")
        try:
            if Decimal(cleaned) * 100 == amount_minor:
                return True
        except InvalidOperation:  # pragma: no cover - the regex yields numbers
            continue
    return False


def check_item(draft: DraftPlanItem, text: str) -> ItemCheck:
    """Decide whether the text backs up one drafted item.

    Args:
        draft: Item drafted by the model.
        text: The pasted text.

    Returns:
        The check outcome (see :class:`ItemCheck`).
    """
    span = find_quote(text, draft.quote)
    if span is None:
        return ItemCheck(
            None, "quote_not_in_text", "the quote is not in the text (or is too short)"
        )
    problem = None
    if "\n" in span or "\r" in span:
        problem = "the quote spans more than one line; quote a single line"
    elif len(span) > QUOTE_MAX_CHARS:
        problem = f"the quote is longer than {QUOTE_MAX_CHARS} characters"
    elif _fold(draft.place_name) not in _fold(span):
        problem = "place_name does not occur in the quote; copy it as written"
    else:
        for label, value in (
            ("start_time", draft.start_time),
            ("end_time", draft.end_time),
        ):
            if value is not None and not _time_in(span, value):
                problem = f"{label} {value} is not written in the quote; leave it empty"
        if draft.amount_minor is not None and not _amount_in(span, draft.amount_minor):
            problem = "amount_minor is not written in the quote; leave it empty"
    if problem is None:
        return ItemCheck(span, None, None)
    return ItemCheck(span, "invalid_item", problem)


def problems(items: Sequence[DraftPlanItem], text: str) -> list[str]:
    """Describe what is wrong with each bad item (to tell the model what to fix).

    Args:
        items: Items drafted by the model.
        text: The pasted text.

    Returns:
        One line per bad item, naming its quote and the problem.
    """
    lines: list[str] = []
    for item in items:
        problem = check_item(item, text).problem
        if problem is not None:
            lines.append(f"- {item.quote[:80]!r}: {problem}")
    return lines


def split_items(
    drafts: Sequence[DraftPlanItem], text: str
) -> tuple[list[ParsedPlanItem], list[UnreadItem]]:
    """Keep the items the text backs up, order them as in the text, list the rest.

    Items come back indexed ``0..n-1`` in the order their quotes appear in the
    text (a stable sort). The kept quote is the span cut from the text, so it
    is an exact substring. Two items with the same quote span count once (the
    first stays). An item that breaks the contract model (bad time, currency,
    ...) is ``invalid_item`` as well.

    Args:
        drafts: Items drafted by the model.
        text: The pasted text.

    Returns:
        ``(items, unread)``.
    """
    found: list[tuple[int, str, DraftPlanItem]] = []
    unread: list[UnreadItem] = []
    seen: set[str] = set()
    for draft in drafts:
        check = check_item(draft, text)
        if check.reason is not None:
            quote = (check.span or draft.quote.strip())[:UNREAD_QUOTE_MAX_CHARS]
            unread.append(UnreadItem(quote=quote, reason=check.reason))
        elif check.span is not None and check.span not in seen:
            seen.add(check.span)
            found.append((text.find(check.span), check.span, draft))
    found.sort(key=itemgetter(0))
    items: list[ParsedPlanItem] = []
    for _, span, draft in found:
        fields = draft.model_dump() | {"quote": span}
        try:
            items.append(ParsedPlanItem(index=len(items), **fields))
        except ValidationError:
            unread.append(
                UnreadItem(quote=span[:UNREAD_QUOTE_MAX_CHARS], reason="invalid_item")
            )
    return items, unread
