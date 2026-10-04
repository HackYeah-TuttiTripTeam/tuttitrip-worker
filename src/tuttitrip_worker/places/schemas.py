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


# --- web research (enrichment) -------------------------------------------------------


class PriceFact(BaseModel):
    """A ticket price the model read on a page."""

    category: str = Field(
        description="adult, child, senior, student, reduced or family."
    )
    amount: float = Field(ge=0)
    currency: str = Field(description="ISO 4217 code, for example EUR.")
    source_url: str = Field(description="The page that states this price.")


class PlaceFacts(BaseModel):
    """What the research model returns for one place; every field is optional.

    Leave a field empty when no page states it. Never guess.
    """

    opening_hours: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Keys mon..sun, values like ['09:30-17:30'] (24h, city-local; a day "
            "that is closed is left out)."
        ),
    )
    hours_source_url: str | None = None
    prices: list[PriceFact] = Field(default_factory=list)
    visit_min: int | None = Field(
        default=None, description="Typical visit length in minutes, from a source."
    )
    indoor: bool | None = None
    child_friendly: bool | None = None
    description: str | None = Field(
        default=None, description="One or two plain sentences."
    )


class ResearchResult(BaseModel):
    """One researched place: the raw facts, the URLs the run cited and its cost."""

    place_id: str
    facts: PlaceFacts | None
    cited_urls: list[str] = Field(default_factory=list)
    cost_usd: float = 0.0


class CleanPrice(BaseModel):
    """A price that passed the checks (stored with ``verified = false``)."""

    category: str
    amount: float
    currency: str
    source_url: str


class CleanFacts(BaseModel):
    """Facts that passed the checks; ``None`` means nothing usable was found."""

    opening_hours: dict[str, Any] | None = None
    hours_source_url: str | None = None
    prices: list[CleanPrice] = Field(default_factory=list)
    visit_min: int | None = None
    indoor: bool | None = None
    child_friendly: bool | None = None
    description: str | None = None


class EnrichTarget(BaseModel):
    """A place picked for research."""

    place_id: str
    name: str
    category: str
    lat: float
    lon: float
    city: str
    country: str | None = None
