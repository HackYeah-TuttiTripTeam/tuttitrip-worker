"""Internal payloads of the places domain (pure: stdlib and Pydantic only).

``PlaceRow`` is one ``places`` row of the OSM import. Its value sets mirror the
backend's catalog enums (``tuttitrip.places.schemas``); the database CHECK
constraints reject anything outside them.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class GeocodedCity(BaseModel):
    """A city found in Nominatim, ready to become a ``cities`` row."""

    model_config = ConfigDict(frozen=True)

    relation_id: int
    name: str = Field(max_length=100)
    country: str = Field(pattern=r"^[A-Z]{2}$")
    timezone: str
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    center_lat: float
    center_lon: float
    bbox_south: float
    bbox_west: float
    bbox_north: float
    bbox_east: float


class RefreshState(BaseModel):
    """What the catalog already knows about a city (read by one step)."""

    model_config = ConfigDict(frozen=True)

    city_name: str | None = None
    country: str | None = None
    timezone: str | None = None
    relation_id: int | None = None
    """OSM relation of the city from the last fetch (saves a Nominatim call)."""
    fresh: bool = False
    """Fetched less than ``refresh_days`` ago: nothing to do."""
    queries_last_day: int = 0
    """Overpass queries this application sent in the last 24 hours."""


class PlaceRow(BaseModel):
    """A catalog place derived from one OSM element (always unverified)."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1, max_length=200)
    category: str
    tags: list[str]
    lat: float
    lon: float
    osm_type: Literal["node", "way", "relation"]
    osm_id: int
    opening_hours: dict[str, Any] | None = None
    wheelchair: bool | None = None
    indoor: bool | None = None
    cuisine: str | None = None
    diet_tags: list[str] = Field(default_factory=list)
    amenities: list[str] = Field(default_factory=list)
