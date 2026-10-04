"""Map one OSM element (Overpass JSON) to a catalog row (pure)."""

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

from tuttitrip_worker.places.constants import (
    CATEGORY_LODGING,
    CATEGORY_RESTAURANT,
    OSM_TYPE_NODE,
)
from tuttitrip_worker.places.logic.hours import weekly_hours
from tuttitrip_worker.places.logic.taxonomy import classify
from tuttitrip_worker.places.schemas import PlaceRow

YES: Final = frozenset({"yes", "only"})

CUISINES: Final[Mapping[str, str]] = {
    "polish": "polish",
    "regional": "polish",
    "italian": "italian",
    "pizza": "italian",
    "pasta": "italian",
    "french": "french",
    "german": "german",
    "british": "british",
    "english": "british",
    "spanish": "spanish",
    "tapas": "spanish",
    "greek": "greek",
    "turkish": "turkish",
    "kebab": "turkish",
    "lebanese": "middle_eastern",
    "arab": "middle_eastern",
    "middle_eastern": "middle_eastern",
    "indian": "indian",
    "chinese": "chinese",
    "japanese": "japanese",
    "sushi": "japanese",
    "ramen": "japanese",
    "thai": "thai",
    "vietnamese": "vietnamese",
    "mexican": "mexican",
    "american": "american",
    "burger": "american",
    "steak_house": "american",
    "international": "international",
}
"""OSM ``cuisine`` values to the catalog's cuisines (others stay unknown)."""

DIETS: Final[Mapping[str, str]] = {
    "diet:vegetarian": "vegetarian",
    "diet:vegan": "vegan",
    "diet:pescetarian": "pescatarian",
    "diet:gluten_free": "gluten_free",
    "diet:lactose_free": "lactose_free",
    "diet:halal": "halal",
    "diet:kosher": "kosher",
}

AMENITIES: Final[Mapping[str, tuple[str, frozenset[str]]]] = {
    "wifi": ("internet_access", frozenset({"wlan", "yes"})),
    "air_conditioning": ("air_conditioning", frozenset({"yes"})),
    "pool": ("swimming_pool", frozenset({"yes"})),
    "pets_allowed": ("dog", frozenset({"yes", "leashed"})),
    "breakfast": ("breakfast", frozenset({"yes"})),
    "elevator": ("elevator", frozenset({"yes"})),
    "wheelchair_accessible": ("wheelchair", frozenset({"yes"})),
}
"""Lodging amenities the catalog checks, from the OSM keys that state them."""

WHEELCHAIR: Final[Mapping[str, bool]] = {"yes": True, "no": False, "limited": False}
"""``limited`` counts as not accessible: the planner must not promise access."""

NAME_MAX = 200


def _cuisine(value: str | None) -> str | None:
    for part in (value or "").split(";"):
        if mapped := CUISINES.get(part.strip().lower()):
            return mapped
    return None


def _point(element: Mapping[str, Any]) -> tuple[float, float] | None:
    source = element if element.get("type") == OSM_TYPE_NODE else element.get("center")
    if not isinstance(source, Mapping):
        return None
    lat, lon = source.get("lat"), source.get("lon")
    if isinstance(lat, int | float) and isinstance(lon, int | float):
        return float(lat), float(lon)
    return None


def to_place_row(
    element: Mapping[str, Any], timezone: str, week_start: date
) -> PlaceRow | None:
    """Turn an Overpass element into a catalog row.

    Args:
        element: One entry of the Overpass ``elements`` array
            (``out center tags``).
        timezone: IANA zone of the city, for the opening hours.
        week_start: Monday of the week the hours are expanded for.

    Returns:
        The row, or ``None`` for an element without a name, a point or a
        matching rule.
    """
    raw_tags = element.get("tags")
    osm_type, osm_id = element.get("type"), element.get("id")
    point = _point(element)
    if (
        not isinstance(raw_tags, Mapping)
        or osm_type not in {"node", "way", "relation"}
        or not isinstance(osm_id, int)
        or point is None
    ):
        return None
    tags: dict[str, str] = {str(k): str(v) for k, v in raw_tags.items()}
    name = tags.get("name", "").strip()[:NAME_MAX]
    found = classify(tags)
    if not name or found is None:
        return None
    lodging = found.category == CATEGORY_LODGING
    hours = tags.get("opening_hours")
    return PlaceRow(
        name=name,
        category=found.category,
        tags=found.tags,
        lat=point[0],
        lon=point[1],
        osm_type=osm_type,
        osm_id=osm_id,
        opening_hours=weekly_hours(hours, timezone, week_start) if hours else None,
        wheelchair=WHEELCHAIR.get(tags.get("wheelchair", "")),
        indoor=found.indoor,
        cuisine=_cuisine(tags.get("cuisine"))
        if found.category == CATEGORY_RESTAURANT
        else None,
        diet_tags=[diet for key, diet in DIETS.items() if tags.get(key) in YES],
        amenities=(
            [
                amenity
                for amenity, (key, accepted) in AMENITIES.items()
                if tags.get(key) in accepted
            ]
            if lodging
            else []
        ),
    )
