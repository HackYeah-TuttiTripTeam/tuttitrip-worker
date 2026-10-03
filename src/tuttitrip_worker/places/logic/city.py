"""Read a Nominatim answer into a city (pure)."""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from babel.numbers import get_territory_currencies
from timezonefinder import TimezoneFinder

from tuttitrip_worker.places.schemas import GeocodedCity

NAME_MAX = 100
AREA_CATEGORIES = frozenset({"boundary", "place"})
ISO_COUNTRY_LENGTH = 2


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
            if item.get("osm_type") == "relation"
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
    currencies = (
        get_territory_currencies(country) if len(country) == ISO_COUNTRY_LENGTH else []
    )
    if zone is None or not currencies:
        return None
    return GeocodedCity(
        relation_id=int(item["osm_id"]),
        name=str(item.get("name") or item["display_name"])[:NAME_MAX],
        country=country,
        timezone=zone,
        currency=currencies[0],
        center_lat=lat,
        center_lon=lon,
        bbox_south=south,
        bbox_west=west,
        bbox_north=north,
        bbox_east=east,
    )
