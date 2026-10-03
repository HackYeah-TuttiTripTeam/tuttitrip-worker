"""Quote checks: a parsed item is only trusted if its quote is in the text.

The model may only read. Each item carries a verbatim quote; an item whose
quote is not a substring of the pasted text (up to white space) was
invented or garbled and goes to the ``unread`` list instead of the result.
Pure: no model, no I/O.
"""

from collections.abc import Sequence
from operator import itemgetter

from pydantic import ValidationError

from tuttitrip_worker.contracts import ParsedPlanItem, UnreadItem
from tuttitrip_worker.linter.schemas import DraftPlanItem
from tuttitrip_worker.quotes import find_quote

QUOTE_MAX_CHARS = 1000
"""Longest quote the contract keeps (``UnreadItem.quote``)."""


def missing_quotes(items: Sequence[DraftPlanItem], text: str) -> list[str]:
    """Quotes of the items that do not occur in the text.

    Args:
        items: Items drafted by the model.
        text: The pasted text.

    Returns:
        The offending quotes in item order (used to tell the model what to fix).
    """
    return [item.quote for item in items if find_quote(text, item.quote) is None]


def split_items(
    drafts: Sequence[DraftPlanItem], text: str
) -> tuple[list[ParsedPlanItem], list[UnreadItem]]:
    """Keep the items with a real quote, order them as in the text, list the rest.

    The kept quote is the span cut from the text, so it is an exact substring.

    Items come back indexed ``0..n-1`` in the order their quotes appear in the
    text (a stable sort, so equal positions keep the model's order). An item
    that has a good quote but breaks the contract model (bad time, currency,
    ...) is ``invalid_item``; one whose quote is absent is ``quote_not_in_text``.

    Args:
        drafts: Items drafted by the model.
        text: The pasted text.

    Returns:
        ``(items, unread)``.
    """
    found: list[tuple[int, str, DraftPlanItem]] = []
    unread: list[UnreadItem] = []
    for draft in drafts:
        span = find_quote(text, draft.quote)
        if span is None:
            unread.append(
                UnreadItem(
                    quote=draft.quote.strip()[:QUOTE_MAX_CHARS],
                    reason="quote_not_in_text",
                )
            )
        else:
            found.append((text.find(span), span, draft))
    found.sort(key=itemgetter(0))
    items: list[ParsedPlanItem] = []
    for _, span, draft in found:
        fields = draft.model_dump() | {"quote": span}
        try:
            items.append(ParsedPlanItem(index=len(items), **fields))
        except ValidationError:
            unread.append(
                UnreadItem(
                    quote=fields["quote"][:QUOTE_MAX_CHARS], reason="invalid_item"
                )
            )
    return items, unread
