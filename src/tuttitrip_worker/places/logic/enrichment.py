"""Checks on what the research model returned (pure).

The rule of the product: a price or opening hours without a source page is not
data. A value is kept only when its ``source_url`` is an http(s) URL that the
search run itself cited or fetched; everything kept is stored unverified.
"""

import re
from typing import Final
from urllib.parse import urlsplit

from tuttitrip_worker.places.schemas import CleanFacts, CleanPrice, PlaceFacts

WEEKDAYS: Final = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
INTERVAL: Final = re.compile(
    r"^\s*(\d{1,2}):(\d{2})\s*[-\u2013]\s*(\d{1,2}):(\d{2})\s*$"
)
CATEGORIES: Final = frozenset(
    {"adult", "child", "senior", "student", "reduced", "family"}
)
MINUTES_IN_DAY: Final = 24 * 60
MINUTES_IN_HOUR: Final = 60
CURRENCY: Final = re.compile(r"^[A-Z]{3}$")
MIN_VISIT_MIN: Final = 10
MAX_VISIT_MIN: Final = 600
MAX_PRICE: Final = 10_000.0
MAX_DESCRIPTION: Final = 400
MAX_URL: Final = 1000


def normalize_url(url: str) -> str:
    """Compare URLs without scheme case, fragment and trailing slash.

    Args:
        url: A URL.

    Returns:
        A comparison key.
    """
    parts = urlsplit(url.strip())
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}?{parts.query}"


def _cited(url: str | None, cited: set[str]) -> str | None:
    if not url or len(url) > MAX_URL:
        return None
    parts = urlsplit(url.strip())
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    return url.strip() if normalize_url(url) in cited else None


def _interval(text: str) -> dict[str, str] | None:
    found = INTERVAL.match(text)
    if not found:
        return None
    opens = int(found[1]) * MINUTES_IN_HOUR + int(found[2])
    closes = int(found[3]) * MINUTES_IN_HOUR + int(found[4])
    if int(found[2]) >= MINUTES_IN_HOUR or int(found[4]) >= MINUTES_IN_HOUR:
        return None
    if not 0 <= opens < closes <= MINUTES_IN_DAY:
        return None
    return {
        "open": f"{opens // 60:02d}:{opens % 60:02d}",
        "close": f"{closes // 60:02d}:{closes % 60:02d}",
    }


def _hours(facts: PlaceFacts) -> dict[str, object] | None:
    if not facts.opening_hours:
        return None
    weekly: dict[str, list[dict[str, str]]] = {}
    for day in WEEKDAYS:
        texts = facts.opening_hours.get(day, [])
        good = [r for text in texts if (r := _interval(text))]
        if len(good) != len(texts):
            return None  # one bad interval: do not store half a week
        if good:
            weekly[day] = good
    return {"weekly": weekly, "closed_dates": []} if weekly else None


def clean(facts: PlaceFacts | None, cited_urls: list[str]) -> CleanFacts:
    """Keep only the facts that are sourced and plausible.

    Args:
        facts: The model's answer, or ``None`` when the run failed.
        cited_urls: URLs the run cited or fetched.

    Returns:
        The checked facts; unsourced hours and prices are dropped.
    """
    if facts is None:
        return CleanFacts()
    cited = {normalize_url(url) for url in cited_urls}
    hours_url = _cited(facts.hours_source_url, cited)
    hours = _hours(facts) if hours_url else None
    prices: dict[str, CleanPrice] = {}
    for price in facts.prices:
        url = _cited(price.source_url, cited)
        currency = price.currency.strip().upper()
        if (
            price.category in CATEGORIES
            and url
            and CURRENCY.match(currency)
            and price.amount <= MAX_PRICE
        ):
            prices.setdefault(
                price.category,
                CleanPrice(
                    category=price.category,
                    amount=round(price.amount, 2),
                    currency=currency,
                    source_url=url,
                ),
            )
    visit = facts.visit_min
    text = (facts.description or "").strip()[:MAX_DESCRIPTION] or None
    return CleanFacts(
        opening_hours=hours,
        hours_source_url=hours_url if hours else None,
        prices=list(prices.values()),
        visit_min=visit if visit and MIN_VISIT_MIN <= visit <= MAX_VISIT_MIN else None,
        indoor=facts.indoor,
        child_friendly=facts.child_friendly,
        description=text,
    )
