"""Read a Nominatim answer into a city (pure)."""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from timezonefinder import TimezoneFinder

from tuttitrip_worker.places.constants import OSM_TYPE_RELATION
from tuttitrip_worker.places.schemas import GeocodedCity

NAME_MAX = 100
AREA_CATEGORIES = frozenset({"boundary", "place"})
# fmt: off
EURO_COUNTRIES = frozenset({
    "AT", "BE", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR", "IE", "IT", "LT",
    "LU", "LV", "MT", "NL", "PT", "SI", "SK", "AD", "MC", "SM", "VA", "ME", "XK",
})
# fmt: on
"""Countries paying in euro (the area plus microstates)."""
OTHER_CURRENCIES = {
    "PL": "PLN", "CZ": "CZK", "HU": "HUF", "RO": "RON", "BG": "BGN", "DK": "DKK",
    "SE": "SEK", "NO": "NOK", "IS": "ISK", "CH": "CHF", "LI": "CHF", "GB": "GBP",
    "UA": "UAH", "RS": "RSD", "BA": "BAM", "AL": "ALL", "MK": "MKD", "MD": "MDL",
    "BY": "BYN", "RU": "RUB", "GE": "GEL", "AM": "AMD", "AZ": "AZN", "TR": "TRY",
    "US": "USD", "CA": "CAD", "MX": "MXN", "BR": "BRL", "AR": "ARS", "CL": "CLP",
    "CO": "COP", "PE": "PEN", "UY": "UYU", "JP": "JPY", "CN": "CNY", "KR": "KRW",
    "IN": "INR", "TH": "THB", "VN": "VND", "ID": "IDR", "MY": "MYR", "SG": "SGD",
    "PH": "PHP", "AU": "AUD", "NZ": "NZD", "AE": "AED", "IL": "ILS", "EG": "EGP",
    "MA": "MAD", "TN": "TND", "ZA": "ZAR", "KE": "KES", "SA": "SAR", "QA": "QAR",
    "JO": "JOD", "HK": "HKD", "TW": "TWD", "CU": "CUP", "DO": "DOP", "CR": "CRC",
}  # fmt: skip
"""Static country to currency table (euro area plus common destinations)."""


@lru_cache(maxsize=1)
def _timezones() -> TimezoneFinder:
    return TimezoneFinder()


def _relation(results: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Find the first boundary relation, the only kind Overpass can turn into an area.

    Args:
        results: Nominatim search results.

    Returns:
        The result, or ``None``.
    """
    return next(
        (
            item
            for item in results
            if item.get("osm_type") == OSM_TYPE_RELATION
            and item.get("category") in AREA_CATEGORIES
        ),
        None,
    )


def parse_city(results: Sequence[Mapping[str, Any]]) -> GeocodedCity | None:
    """Pick the city boundary from a Nominatim ``jsonv2`` search answer.

    Args:
        results: The decoded JSON list (``addressdetails=1``).

    Returns:
        The city with its zone and currency, or ``None`` when the answer has no
        usable boundary relation or country.
    """
    item = _relation(results)
    address = item.get("address") if item else None
    if item is None or not isinstance(address, Mapping):
        return None
    country = str(address.get("country_code", "")).upper()
    south, north, west, east = (float(value) for value in item["boundingbox"])
    lat, lon = float(item["lat"]), float(item["lon"])
    zone = _timezones().timezone_at(lat=lat, lng=lon)
    currency = "EUR" if country in EURO_COUNTRIES else OTHER_CURRENCIES.get(country)
    if zone is None or currency is None:
        return None
    return GeocodedCity(
        relation_id=int(item["osm_id"]),
        name=str(item.get("name") or item["display_name"])[:NAME_MAX],
        country=country,
        timezone=zone,
        currency=currency,
        center_lat=lat,
        center_lon=lon,
        bbox_south=south,
        bbox_west=west,
        bbox_north=north,
        bbox_east=east,
    )
