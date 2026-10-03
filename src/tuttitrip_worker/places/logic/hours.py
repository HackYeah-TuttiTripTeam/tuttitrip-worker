"""OSM ``opening_hours`` to the catalog's weekly format (pure)."""

from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from opening_hours import OpeningHours, State

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
MAX_EXPRESSION_LENGTH = 500


def monday_of(now: datetime, timezone: str) -> date:
    """Monday of the week that contains a moment, in the city's zone.

    Args:
        now: An aware moment.
        timezone: IANA zone of the city.

    Returns:
        The date of that Monday.
    """
    local = now.astimezone(ZoneInfo(timezone)).date()
    return local - timedelta(days=local.weekday())


def _clock(moment: datetime, midnight_after: datetime) -> str:
    return "24:00" if moment >= midnight_after else moment.strftime("%H:%M")


def _expand(
    expression: str, timezone: str, week_start: date
) -> dict[str, list[dict[str, str]]]:
    zone = ZoneInfo(timezone)
    parsed = OpeningHours(expression, timezone=zone)
    weekly: dict[str, list[dict[str, str]]] = {}
    for offset, weekday in enumerate(WEEKDAYS):
        start = datetime.combine(week_start + timedelta(days=offset), time(), zone)
        end = datetime.combine(week_start + timedelta(days=offset + 1), time(), zone)
        ranges = [
            {"open": opens.strftime("%H:%M"), "close": _clock(closes, end)}
            for opens, closes, state, _comment in parsed.intervals(start, end)
            if state is State.OPEN
        ]
        if ranges:
            weekly[weekday] = ranges
    return weekly


def weekly_hours(
    expression: str, timezone: str, week_start: date
) -> dict[str, Any] | None:
    """Convert an OSM ``opening_hours`` expression to the weekly format.

    The week of ``week_start`` (a Monday) is expanded and read back per day, so
    seasonal rules and ``PH`` (public holidays, unknown without a region)
    resolve as they would in that week. Intervals over midnight are split at
    midnight, as the catalog requires.

    Args:
        expression: Value of the OSM ``opening_hours`` tag.
        timezone: IANA zone of the city (the times stay local).
        week_start: The Monday that stands for a typical week.

    Returns:
        ``{"weekly": {"mon": [{"open": "09:00", "close": "17:00"}], ...},
        "closed_dates": []}``, or ``None`` when the expression cannot be
        parsed or is never open.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        return None
    try:
        weekly = _expand(expression, timezone, week_start)
    except Exception:  # ruff: ignore[blind-except] the library documents no exception types
        return None
    return {"weekly": weekly, "closed_dates": []} if weekly else None
