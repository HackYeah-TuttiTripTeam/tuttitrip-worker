"""Rules from OSM tags to catalog categories and interest tags (pure).

One table drives both the Overpass query (:func:`overpass_query`) and the
mapping of an element (:func:`classify`), so they cannot drift apart. Values
are the backend's catalog vocabulary (``PlaceCategory``, ``PlaceTag``).
"""

from collections.abc import Mapping
from itertools import groupby
from typing import NamedTuple

from tuttitrip_worker.places.constants import OSM_AREA_ID_OFFSET

QUERY_TIMEOUT_SEC = 90
"""Server-side timeout of the Overpass query."""


class Rule(NamedTuple):
    """An OSM ``key=value`` that makes an element a candidate."""

    key: str
    value: str
    category: str
    tags: tuple[str, ...] = ()
    indoor: bool | None = None
    needs: str | None = None
    """Another key the element must have (keeps places of worship to notable ones)."""


_LODGING = ("hotel", "hostel", "guest_house", "apartment", "motel", "chalet")

RULES: tuple[Rule, ...] = (
    *(Rule("tourism", value, "lodging") for value in _LODGING),
    Rule("tourism", "museum", "museum", ("museums",), indoor=True),
    Rule("tourism", "gallery", "museum", ("art", "museums"), indoor=True),
    Rule("tourism", "attraction", "attraction"),
    Rule("tourism", "zoo", "attraction", ("animals", "family", "kids"), indoor=False),
    Rule(
        "tourism", "aquarium", "attraction", ("animals", "water", "family"), indoor=True
    ),
    Rule(
        "tourism",
        "theme_park",
        "entertainment",
        ("family", "kids", "adventure"),
        indoor=False,
    ),
    Rule("tourism", "viewpoint", "viewpoint", ("views",), indoor=False),
    Rule("historic", "castle", "attraction", ("history", "architecture")),
    Rule("historic", "fort", "attraction", ("history", "architecture")),
    Rule("historic", "manor", "attraction", ("history", "architecture")),
    Rule("historic", "city_gate", "attraction", ("history", "architecture")),
    Rule("historic", "monument", "attraction", ("history",)),
    Rule("historic", "memorial", "attraction", ("history",)),
    Rule("historic", "ruins", "attraction", ("history",)),
    Rule("historic", "archaeological_site", "attraction", ("history",)),
    Rule(
        "amenity",
        "place_of_worship",
        "attraction",
        ("religion", "architecture"),
        needs="wikidata",
    ),
    Rule("amenity", "restaurant", "restaurant"),
    Rule("amenity", "fast_food", "restaurant", ("street_food",)),
    Rule("amenity", "food_court", "restaurant", ("street_food",)),
    Rule("amenity", "cafe", "cafe"),
    Rule("amenity", "ice_cream", "cafe"),
    Rule("amenity", "bar", "nightlife", ("nightlife",)),
    Rule("amenity", "pub", "nightlife", ("nightlife",)),
    Rule("amenity", "nightclub", "nightlife", ("nightlife", "music")),
    Rule("amenity", "cinema", "entertainment", indoor=True),
    Rule("amenity", "theatre", "entertainment", ("art",), indoor=True),
    Rule("amenity", "arts_centre", "entertainment", ("art",), indoor=True),
    Rule("amenity", "marketplace", "shopping", ("markets", "local_food")),
    Rule("leisure", "park", "park", ("parks", "nature"), indoor=False),
    Rule("leisure", "garden", "park", ("parks", "nature"), indoor=False),
    Rule("leisure", "nature_reserve", "park", ("nature",), indoor=False),
    Rule(
        "leisure", "playground", "park", ("playground", "kids", "family"), indoor=False
    ),
    Rule(
        "leisure",
        "water_park",
        "entertainment",
        ("water", "family", "kids"),
        indoor=False,
    ),
    Rule("natural", "beach", "park", ("beaches", "water"), indoor=False),
    Rule("shop", "mall", "shopping", ("shopping",), indoor=True),
    Rule("shop", "department_store", "shopping", ("shopping",), indoor=True),
)


class Classification(NamedTuple):
    """Result of :func:`classify`."""

    category: str
    tags: list[str]
    indoor: bool | None


def overpass_query(relation_id: int) -> str:
    """Build the single Overpass query for a city relation.

    Only named elements are requested; ``out center tags`` gives one point for
    ways and relations.

    Args:
        relation_id: OSM relation id of the city boundary.

    Returns:
        Overpass QL text, in the form ``nwr[...](area); out center tags;``.
    """
    clauses: list[str] = []
    ordered = sorted(RULES, key=lambda rule: (rule.key, rule.needs or ""))
    for (key, needs), group in groupby(ordered, lambda r: (r.key, r.needs or "")):
        values = "|".join(sorted({rule.value for rule in group}))
        extra = f'["{needs}"]' if needs else ""
        clauses.append(f'nwr["{key}"~"^({values})$"]["name"]{extra}(area.city);')
    body = "\n  ".join(clauses)
    return (
        f"[out:json][timeout:{QUERY_TIMEOUT_SEC}];\n"
        f"area({OSM_AREA_ID_OFFSET + relation_id})->.city;\n"
        f"(\n  {body}\n);\n"
        "out center tags;"
    )


def classify(osm_tags: Mapping[str, str]) -> Classification | None:
    """Pick the catalog category and interest tags of an element.

    The first matching rule (in :data:`RULES` order) decides the category;
    tags of every matching rule are merged. An explicit ``indoor`` tag wins
    over the rule.

    Args:
        osm_tags: Tags of the OSM element.

    Returns:
        The classification, or ``None`` when no rule matches.
    """
    matches = [
        rule
        for rule in RULES
        if osm_tags.get(rule.key) == rule.value
        and (rule.needs is None or rule.needs in osm_tags)
    ]
    if not matches:
        return None
    tags = list(dict.fromkeys(tag for rule in matches for tag in rule.tags))
    explicit = {"yes": True, "no": False}.get(osm_tags.get("indoor", ""))
    indoor = explicit if explicit is not None else matches[0].indoor
    return Classification(matches[0].category, tags, indoor)
