"""Quote checks: a parsed item is only trusted if its quote is in the text.

The model may only read. Each item carries a verbatim quote; an item whose
quote is not a substring of the pasted text (after collapsing white space) was
invented or garbled and goes to the ``unread`` list instead of the result.
Pure: no model, no I/O.
"""

from collections.abc import Sequence
from operator import itemgetter

from pydantic import ValidationError

from tuttitrip_worker.contracts import ParsedPlanItem, UnreadItem
from tuttitrip_worker.linter.schemas import DraftPlanItem

QUOTE_MAX_CHARS = 1000
"""Longest quote the contract keeps (``UnreadItem.quote``)."""


def normalize_whitespace(text: str) -> str:
    """Collapse every run of white space (newlines, tabs, nbsp) to one space.

    Args:
        text: Pasted text or a quote.

    Returns:
        The text with single spaces and no leading or trailing white space.
    """
    return " ".join(text.split())


def quote_in_text(quote: str, text: str) -> bool:
    """Whether the quote occurs in the text, ignoring white-space differences.

    Letters, case and punctuation must match exactly. A blank quote never
    matches (it would be a substring of everything).

    Args:
        quote: Quote returned by the model.
        text: The pasted text.

    Returns:
        ``True`` when the normalized quote is a substring of the normalized text.
    """
    needle = normalize_whitespace(quote)
    return bool(needle) and needle in normalize_whitespace(text)


def missing_quotes(items: Sequence[DraftPlanItem], text: str) -> list[str]:
    """Quotes of the items that do not occur in the text.

    Args:
        items: Items drafted by the model.
        text: The pasted text.

    Returns:
        The offending quotes in item order (used to tell the model what to fix).
    """
    return [item.quote for item in items if not quote_in_text(item.quote, text)]


def split_items(
    drafts: Sequence[DraftPlanItem], text: str
) -> tuple[list[ParsedPlanItem], list[UnreadItem]]:
    """Keep the items with a real quote, order them as in the text, list the rest.

    The quote is kept as the model wrote it (stripped), so a consumer applies
    the same white-space rule as :func:`quote_in_text`.

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
    normalized = normalize_whitespace(text)
    found: list[tuple[int, DraftPlanItem]] = []
    unread: list[UnreadItem] = []
    for draft in drafts:
        needle = normalize_whitespace(draft.quote)
        position = normalized.find(needle) if needle else -1
        if position < 0:
            unread.append(
                UnreadItem(quote=needle[:QUOTE_MAX_CHARS], reason="quote_not_in_text")
            )
        else:
            found.append((position, draft))
    found.sort(key=itemgetter(0))
    items: list[ParsedPlanItem] = []
    for _, draft in found:
        fields = draft.model_dump() | {"quote": draft.quote.strip()}
        try:
            items.append(ParsedPlanItem(index=len(items), **fields))
        except ValidationError:
            unread.append(
                UnreadItem(
                    quote=fields["quote"][:QUOTE_MAX_CHARS], reason="invalid_item"
                )
            )
    return items, unread
