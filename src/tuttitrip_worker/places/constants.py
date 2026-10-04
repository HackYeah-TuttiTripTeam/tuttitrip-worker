"""Fixed values of ``fetch_place_candidates``."""

from typing import Final

PROGRESS_LOCATING: Final = ("locating", 10)
"""``progress`` event while the city is looked up: ``(stage, percent)``."""

PROGRESS_FETCHING: Final = ("fetching", 40)
"""``progress`` event while Overpass is queried."""

OSM_AREA_ID_OFFSET: Final = 3_600_000_000
"""Overpass area id = this offset + the OSM relation id."""

OSM_TYPE_RELATION: Final = "relation"
"""Nominatim ``osm_type`` of an administrative area."""

OSM_TYPE_NODE: Final = "node"
"""Overpass element type that carries ``lat`` and ``lon`` itself."""

SOURCE_OSM: Final = "osm"
"""``places.source`` of rows written from OpenStreetMap."""

CATEGORY_LODGING: Final = "lodging"
"""Place category of accommodation."""

CATEGORY_RESTAURANT: Final = "restaurant"
"""Place category of restaurants (the only one with a cuisine)."""

OVERPASS_BUSY_MARKER: Final = "runtime error"
"""Text in an Overpass ``remark`` that means the server is overloaded."""
