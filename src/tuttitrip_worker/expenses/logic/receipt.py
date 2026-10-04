"""Rules S1 for a receipt reading (pure): is the reading consistent?

The model only reads and proposes a category; these rules decide whether the
host has to confirm the draft. Every rule that fires adds a machine-readable
reason, and any reason sets ``needs_confirmation``.
"""

import re
from datetime import date
from typing import Final

from tuttitrip_worker.contracts import CURRENCY_PATTERN, ReadReceiptOutput
from tuttitrip_worker.expenses.schemas import Judgement, ReceiptReading, TripDates

CONFIDENCE_FLOOR: Final = 0.5
"""Below this margin (0..1) the decision model is not trusted; tune on real data."""

KNOWN_CURRENCIES: Final = frozenset(
    {
        *("PLN", "EUR", "USD", "GBP", "CHF", "CZK", "HUF", "SEK", "NOK"),
        *("DKK", "RON", "BGN", "TRY", "UAH", "CAD", "AUD"),
    }
)
"""Currencies the settlement can convert (NBP tables A and B, two decimals)."""

MAX_REASONS: Final = 10


def parse_date(value: str | None) -> date | None:
    """Read ``YYYY-MM-DD``.

    Args:
        value: Text from the model.

    Returns:
        The date, or ``None`` when it is missing or not a real date.
    """
    try:
        return date.fromisoformat(value) if value else None
    except ValueError:
        return None


def clean_currency(value: str | None) -> str | None:
    """Upper-case an ISO 4217 code and drop anything that is not one.

    Args:
        value: Text from the model.

    Returns:
        Three capital letters, or ``None``.
    """
    code = (value or "").strip().upper()
    return code if re.fullmatch(CURRENCY_PATTERN, code) else None


def reasons_for(
    reading: ReceiptReading, trip: TripDates, judgement: Judgement
) -> list[str]:
    """Collect why the host should confirm the reading.

    Args:
        reading: What the vision model read.
        trip: Dates of the trip.
        judgement: Category decision with its confidence.

    Returns:
        Reason codes in a fixed order; empty means no confirmation is needed.
    """
    reasons: list[str] = []
    items_total = sum(line.amount_minor for line in reading.items)
    if reading.items and items_total != reading.amount_minor:
        reasons.append("items_sum_mismatch")
    if reading.currency is None:
        reasons.append("currency_missing")
    elif reading.currency not in KNOWN_CURRENCIES:
        reasons.append("currency_unknown")
    spent_on = parse_date(reading.spent_on)
    if reading.spent_on and spent_on is None:
        reasons.append("date_invalid")
    elif spent_on is not None and (
        (trip.start is not None and spent_on < trip.start)
        or (trip.end is not None and spent_on > trip.end)
    ):
        reasons.append("date_outside_trip")
    if judgement.category is None:
        reasons.append("category_unavailable")
    elif judgement.confidence is None or judgement.confidence < CONFIDENCE_FLOOR:
        reasons.append("low_confidence")
    if judgement.needs_confirmation:
        reasons.append("model_asks_confirmation")
    return reasons[:MAX_REASONS]


def build_output(
    reading: ReceiptReading, trip: TripDates, judgement: Judgement
) -> ReadReceiptOutput | None:
    """Turn a reading into the contract output.

    Args:
        reading: What the vision model read.
        trip: Dates of the trip.
        judgement: Category decision with its confidence.

    Returns:
        The output, or ``None`` when no positive amount was read (it is never
        guessed, so the job fails instead).
    """
    amount = reading.amount_minor
    if amount is None or amount <= 0:
        return None
    reading = reading.model_copy(update={"currency": clean_currency(reading.currency)})
    reasons = reasons_for(reading, trip, judgement)
    spent_on = parse_date(reading.spent_on)
    return ReadReceiptOutput(
        amount_minor=amount,
        currency=reading.currency,
        spent_on=spent_on.isoformat() if spent_on else None,
        merchant=(reading.merchant or "")[:200] or None,
        category=judgement.category,
        needs_confirmation=bool(reasons),
        reasons=reasons,
    )
