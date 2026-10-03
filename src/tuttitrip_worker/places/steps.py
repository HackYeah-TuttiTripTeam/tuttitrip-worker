"""I/O steps of the places domain: Nominatim, Overpass and the catalog.

Usage policies, kept here (see ``OsmSettings``):

* Overpass: at most one query at a time per process (``_gate``), a 30 s pause
  before every retry (``RETRY_PAUSE_SEC``, the DBOS retry interval), at most
  ``overpass_daily_limit`` queries a day, a ``User-Agent`` that names TuttiTrip.
* Nominatim: one request per workflow, the workflow pauses 1 s around it; the
  answer is cached for good as the ``cities`` row and the refresh marker.

Every step is idempotent: the city insert does nothing on conflict, places are
upserted by (``osm_type``, ``osm_id``) and never overwrite sheet rows.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import httpx
from dbos import DBOS
from sqlalchemy import DateTime, Select, case, cast, func, select
from sqlalchemy.dialects.postgresql import Insert, insert
from sqlalchemy.engine import RowMapping

from tuttitrip_worker.contracts import Workflow
from tuttitrip_worker.places.logic.city import parse_city
from tuttitrip_worker.places.logic.hours import monday_of
from tuttitrip_worker.places.logic.mapping import to_place_row
from tuttitrip_worker.places.logic.taxonomy import overpass_query
from tuttitrip_worker.places.schemas import GeocodedCity, PlaceRow, RefreshState
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.job_results import build_job_result_upsert
from tuttitrip_worker.shared.db.tables import cities, job_results, places

RETRY_PAUSE_SEC: Final = 30.0
"""Pause before the first retry after a 429 (Overpass policy); it doubles."""
MAX_ATTEMPTS: Final = 4
HTTP_TIMEOUT_SEC: Final = 120.0
BATCH_SIZE: Final = 500
MARKER_PREFIX: Final = "osm-fetch:"
"""``job_results.workflow_id`` prefix of the per-city refresh marker."""
SERVER_ERRORS: Final = 500

_gate = asyncio.Lock()
"""One OSM request at a time in this process (no parallel queries)."""


class OsmBusyError(RuntimeError):
    """Nominatim or Overpass asked us to slow down (429, 5xx, query timed out)."""


def _is_transient(error: BaseException) -> bool:
    return isinstance(error, OsmBusyError | httpx.TransportError)


def build_client() -> httpx.AsyncClient:
    """HTTP client that names the application in every request.

    Returns:
        A client with the configured ``User-Agent`` and timeout.
    """
    return httpx.AsyncClient(
        headers={"User-Agent": get_settings().osm.user_agent},
        timeout=HTTP_TIMEOUT_SEC,
    )


def _check(response: httpx.Response) -> None:
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        msg = f"{response.url.host} answered 429"
        raise OsmBusyError(msg)
    if response.status_code >= SERVER_ERRORS:
        msg = f"{response.url.host} answered {response.status_code}"
        raise OsmBusyError(msg)
    response.raise_for_status()


def marker_id(slug: str) -> str:
    """Key of the refresh marker of a city in ``job_results``.

    Args:
        slug: City slug.

    Returns:
        The ``workflow_id`` of the marker row.
    """
    return f"{MARKER_PREFIX}{slug}"


# --- statements (pure builders, compiled in tests) --------------------------------


def build_city_select(slug: str) -> Select[Any, Any, Any]:
    """``SELECT name, country, timezone`` of one city.

    Args:
        slug: City slug.

    Returns:
        The statement (not executed).
    """
    return select(cities.c.name, cities.c.country, cities.c.timezone).where(
        cities.c.slug == slug
    )


def build_marker_select(slug: str) -> Select[dict[str, Any]]:
    """``SELECT result`` of the refresh marker of one city.

    Args:
        slug: City slug.

    Returns:
        The statement (not executed).
    """
    return select(job_results.c.result).where(
        job_results.c.workflow_id == marker_id(slug)
    )


def build_recent_queries_select(since: datetime) -> Select[int]:
    """Count the Overpass queries (markers written) since a moment.

    Args:
        since: Start of the window.

    Returns:
        The statement (not executed).
    """
    fetched_at = cast(
        job_results.c.result["fetched_at"].astext, DateTime(timezone=True)
    )
    return select(func.count()).where(
        job_results.c.workflow_id.like(f"{MARKER_PREFIX}%"), fetched_at > since
    )


def build_city_insert(slug: str, city: GeocodedCity) -> Insert:
    """Insert a city; an existing one (for example from the sheet) stays as is.

    Args:
        slug: City slug.
        city: Geocoded boundary.

    Returns:
        The statement (not executed).
    """
    values = city.model_dump(exclude={"relation_id"})
    return (
        insert(cities)
        .values(slug=slug, **values)
        .on_conflict_do_nothing(index_elements=[cities.c.slug])
    )


def place_values(slug: str, row: PlaceRow) -> dict[str, Any]:
    """Column values of a new, unverified OSM place.

    Args:
        slug: City the place belongs to.
        row: Mapped OSM element.

    Returns:
        Values for ``INSERT`` (the id and the other columns use server defaults).
    """
    return {
        **row.model_dump(),
        "city_slug": slug,
        "source": "osm",
        "hours_verified": False,
    }


def build_places_upsert(slug: str, rows: list[PlaceRow]) -> Insert:
    """Upsert places by (``osm_type``, ``osm_id``).

    Only rows with ``source = 'osm'`` are updated, so a sheet row is never
    overwritten, and hours a person verified are kept.

    Args:
        slug: City the places belong to.
        rows: Mapped OSM elements, unique by (``osm_type``, ``osm_id``).

    Returns:
        The statement (not executed).
    """
    statement = insert(places).values([place_values(slug, row) for row in rows])
    excluded = statement.excluded
    return statement.on_conflict_do_update(
        index_elements=[places.c.osm_type, places.c.osm_id],
        set_={
            "name": excluded.name,
            "category": excluded.category,
            "tags": excluded.tags,
            "lat": excluded.lat,
            "lon": excluded.lon,
            "wheelchair": excluded.wheelchair,
            "indoor": excluded.indoor,
            "cuisine": excluded.cuisine,
            "diet_tags": excluded.diet_tags,
            "amenities": excluded.amenities,
            "opening_hours": case(
                (places.c.hours_verified, places.c.opening_hours),
                else_=excluded.opening_hours,
            ),
        },
        where=places.c.source == "osm",
    )


# --- steps -----------------------------------------------------------------------


def _marker_relation(result: RowMapping | None) -> tuple[int | None, datetime | None]:
    if result is None:
        return None, None
    data = result["result"]
    return int(data["relation_id"]), datetime.fromisoformat(str(data["fetched_at"]))


@DBOS.step(retries_allowed=True, max_attempts=3)
async def load_refresh_state(slug: str) -> dict[str, Any]:
    """Read what the catalog knows about a city and how much quota is left.

    Args:
        slug: City slug.

    Returns:
        A ``RefreshState`` as JSON.
    """
    settings = get_settings().osm
    now = datetime.now(UTC)
    async with transaction() as connection:
        city = (await connection.execute(build_city_select(slug))).mappings().first()
        marker = (
            (await connection.execute(build_marker_select(slug))).mappings().first()
        )
        recent = (
            await connection.execute(
                build_recent_queries_select(now - timedelta(days=1))
            )
        ).scalar_one()
    relation_id, fetched_at = _marker_relation(marker)
    fresh = fetched_at is not None and now - fetched_at < timedelta(
        days=settings.refresh_days
    )
    return RefreshState(
        city_name=city["name"] if city else None,
        country=city["country"] if city else None,
        timezone=city["timezone"] if city else None,
        relation_id=relation_id,
        fresh=fresh,
        queries_last_day=int(recent),
    ).model_dump(mode="json")


@DBOS.step(
    retries_allowed=True,
    max_attempts=MAX_ATTEMPTS,
    interval_seconds=RETRY_PAUSE_SEC,
    should_retry=_is_transient,
)
async def geocode_city(query: str, country: str | None) -> dict[str, Any] | None:
    """Find the city boundary in Nominatim (one request).

    Args:
        query: Free-text city name.
        country: ISO 3166-1 alpha-2 code to narrow the search, if known.

    Returns:
        A ``GeocodedCity`` as JSON, or ``None`` when Nominatim knows no such city.
    """
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": "1",
        "featuretype": "settlement",
        "limit": "5",
    }
    if country:
        params["countrycodes"] = country.lower()
    async with _gate, build_client() as client:
        response = await client.get(
            f"{get_settings().osm.nominatim_url}/search", params=params
        )
    _check(response)
    city = parse_city(response.json())
    return None if city is None else city.model_dump(mode="json")


@DBOS.step(retries_allowed=True, max_attempts=3)
async def store_city(slug: str, city: dict[str, Any]) -> None:
    """Insert the city unless the catalog already has it.

    Args:
        slug: City slug.
        city: A ``GeocodedCity`` as JSON.
    """
    statement = build_city_insert(slug, GeocodedCity.model_validate(city))
    async with transaction() as connection:
        await connection.execute(statement)


@DBOS.step(
    retries_allowed=True,
    max_attempts=MAX_ATTEMPTS,
    interval_seconds=RETRY_PAUSE_SEC,
    should_retry=_is_transient,
)
async def import_places(slug: str, relation_id: int, timezone: str) -> int:
    """Run the one Overpass query of the city and upsert the places.

    A crash after the query repeats it on recovery; the upsert makes that safe.

    Args:
        slug: City slug (the city row must exist).
        relation_id: OSM relation of the city boundary.
        timezone: IANA zone of the city, for the opening hours.

    Returns:
        How many places were written.
    """
    async with _gate, build_client() as client:
        response = await client.post(
            get_settings().osm.overpass_url, data={"data": overpass_query(relation_id)}
        )
    _check(response)
    body = response.json()
    remark = str(body.get("remark", ""))
    if "runtime error" in remark:
        raise OsmBusyError(remark)
    week = monday_of(datetime.now(UTC), timezone)
    mapped = (to_place_row(element, timezone, week) for element in body["elements"])
    unique = {(row.osm_type, row.osm_id): row for row in mapped if row is not None}
    rows = list(unique.values())
    async with transaction() as connection:
        for start in range(0, len(rows), BATCH_SIZE):
            await connection.execute(
                build_places_upsert(slug, rows[start : start + BATCH_SIZE])
            )
    return len(rows)


@DBOS.step(retries_allowed=True, max_attempts=3)
async def mark_fetched(slug: str, relation_id: int, stored: int) -> None:
    """Record the fetch: it starts the refresh period and counts toward the quota.

    Args:
        slug: City slug.
        relation_id: OSM relation that was queried (saves a Nominatim call later).
        stored: How many places were written.
    """
    result = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "relation_id": relation_id,
        "stored": stored,
    }
    statement = build_job_result_upsert(
        marker_id(slug), Workflow.FETCH_PLACE_CANDIDATES.value, result
    )
    async with transaction() as connection:
        await connection.execute(statement)
