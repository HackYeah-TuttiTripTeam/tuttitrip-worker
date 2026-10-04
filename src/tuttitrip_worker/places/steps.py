"""I/O steps of the places domain: Nominatim, Overpass and the catalog.

Usage policies, kept here (see ``OsmSettings``):

* One OSM request at a time per process (``_gate``), and the gate is held for
  1 s after every request (``REQUEST_SPACING_SEC``), so Nominatim's 1 request
  a second holds across workflows. A ``User-Agent`` names TuttiTrip.
* Overpass: a pause of at least 30 s before every retry (``RETRY_PAUSE_SEC``,
  the DBOS retry interval; a longer ``Retry-After`` is waited out too). Each
  workflow reserves a slot (``reserve_overpass_slot``) before its first
  Overpass call; at most ``overpass_daily_limit`` reservations a day, failed
  attempts included.
* The Nominatim answer is cached as the ``cities`` row; the fetch state is
  ``city_fetches`` (refresh period, OSM relation) and ``city_fetch_attempts``
  (daily quota), both owned by the backend.

Every step is idempotent: the city insert does nothing on conflict, places are
upserted by (``osm_type``, ``osm_id``) and never overwrite sheet rows.
"""

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, Final

import httpx
from dbos import DBOS
from sqlalchemy import Select, case, func, select
from sqlalchemy.dialects.postgresql import Insert, insert

from tuttitrip_worker.places.logic.city import parse_city
from tuttitrip_worker.places.logic.hours import monday_of
from tuttitrip_worker.places.logic.mapping import to_place_row
from tuttitrip_worker.places.logic.taxonomy import overpass_query
from tuttitrip_worker.places.schemas import GeocodedCity, PlaceRow, RefreshState
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import (
    cities,
    city_fetch_attempts,
    city_fetches,
    places,
)

RETRY_PAUSE_SEC: Final = 30.0
"""Pause before the first retry after a 429 (Overpass policy); it doubles."""
MAX_ATTEMPTS: Final = 4
HTTP_TIMEOUT_SEC: Final = 120.0
BATCH_SIZE: Final = 500
LOCK_KEY: Final = 0x0054_7574_5472_6970
"""Key of the advisory lock that serializes reservations."""
SERVER_ERRORS: Final = 500
REQUEST_SPACING_SEC: Final = 1.0
"""The gate stays closed this long after every request (Nominatim: 1 a second)."""
MAX_RETRY_AFTER_SEC: Final = 300.0

_gate = asyncio.Lock()
"""One OSM request at a time in this process (no parallel queries)."""


class OsmBusyError(RuntimeError):
    """Nominatim or Overpass asked us to slow down (429, 5xx, query timed out)."""


_sleep = asyncio.sleep
"""Indirection so tests do not really wait."""


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


def _retry_after(response: httpx.Response) -> float:
    try:
        return min(float(response.headers.get("Retry-After", 0)), MAX_RETRY_AFTER_SEC)
    except ValueError:  # an HTTP date: the default pause is enough
        return 0.0


async def _send(
    send: Callable[[httpx.AsyncClient], Awaitable[httpx.Response]],
) -> httpx.Response:
    """Send one request through the gate, then keep the gate closed for a second.

    On 429 the rest of ``Retry-After`` beyond the DBOS retry pause is waited out
    here, still holding the gate, so the retry comes after ``max(30 s, Retry-After)``.

    Args:
        send: Makes the request with the given client.

    Returns:
        A successful response.

    Raises:
        OsmBusyError: On 429, a 5xx answer or a timed out query.
    """
    async with _gate:
        try:
            async with build_client() as client:
                response = await send(client)
            if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
                await _sleep(max(0.0, _retry_after(response) - RETRY_PAUSE_SEC))
        finally:
            await _sleep(REQUEST_SPACING_SEC)
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        msg = f"{response.url.host} answered 429"
        raise OsmBusyError(msg)
    if response.status_code >= SERVER_ERRORS:
        msg = f"{response.url.host} answered {response.status_code}"
        raise OsmBusyError(msg)
    response.raise_for_status()
    return response


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


def build_fetch_select(slug: str) -> Select[Any, Any]:
    """``SELECT osm_relation_id, fetched_at`` of the last fetch of one city.

    Args:
        slug: City slug.

    Returns:
        The statement (not executed).
    """
    return select(city_fetches.c.osm_relation_id, city_fetches.c.fetched_at).where(
        city_fetches.c.city_slug == slug
    )


def build_recent_attempts_select(since: datetime) -> Select[int]:
    """Count the reservations (attempts, failed ones too) since a moment.

    Args:
        since: Start of the window.

    Returns:
        The statement (not executed).
    """
    return select(func.count()).where(city_fetch_attempts.c.reserved_at > since)


def build_attempt_select(workflow_id: str) -> Select[int]:
    """Count the reservation rows of one workflow (0 or 1).

    Args:
        workflow_id: DBOS id of the workflow.

    Returns:
        The statement (not executed).
    """
    return select(func.count()).where(city_fetch_attempts.c.workflow_id == workflow_id)


def build_attempt_insert(workflow_id: str, slug: str, now: datetime) -> Insert:
    """Insert the reservation row of a workflow; a repeat changes nothing.

    Args:
        workflow_id: DBOS id of the workflow.
        slug: City slug, for the record.
        now: Moment of the reservation.

    Returns:
        The statement (not executed).
    """
    return (
        insert(city_fetch_attempts)
        .values(workflow_id=workflow_id, city_slug=slug, reserved_at=now)
        .on_conflict_do_nothing(index_elements=[city_fetch_attempts.c.workflow_id])
    )


def build_fetch_upsert(
    slug: str, relation_id: int, stored: int, now: datetime
) -> Insert:
    """Upsert the last fetch of a city (the city row must exist: foreign key).

    Args:
        slug: City slug.
        relation_id: OSM relation that was queried.
        stored: How many places were written.
        now: Moment of the fetch.

    Returns:
        The statement (not executed).
    """
    statement = insert(city_fetches).values(
        city_slug=slug, osm_relation_id=relation_id, fetched_at=now, stored=stored
    )
    return statement.on_conflict_do_update(
        index_elements=[city_fetches.c.city_slug],
        set_={
            "osm_relation_id": statement.excluded.osm_relation_id,
            "fetched_at": statement.excluded.fetched_at,
            "stored": statement.excluded.stored,
        },
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


@DBOS.step(retries_allowed=True, max_attempts=3)
async def load_refresh_state(slug: str) -> dict[str, Any]:
    """Read what the catalog knows about a city.

    Args:
        slug: City slug.

    Returns:
        A ``RefreshState`` as JSON.
    """
    settings = get_settings().osm
    now = datetime.now(UTC)
    async with transaction() as connection:
        city = (await connection.execute(build_city_select(slug))).mappings().first()
        fetch = (await connection.execute(build_fetch_select(slug))).mappings().first()
    relation_id = int(fetch["osm_relation_id"]) if fetch else None
    fresh = fetch is not None and now - fetch["fetched_at"] < timedelta(
        days=settings.refresh_days
    )
    return RefreshState(
        city_name=city["name"] if city else None,
        country=city["country"] if city else None,
        timezone=city["timezone"] if city else None,
        relation_id=relation_id,
        fresh=fresh,
    ).model_dump(mode="json")


@DBOS.step(retries_allowed=True, max_attempts=3)
async def reserve_overpass_slot(workflow_id: str, slug: str) -> bool:
    """Reserve one of today's Overpass queries for this workflow.

    One transaction under an advisory lock: count the reservations of the last
    24 hours (failed attempts included), compare with the limit and insert this
    workflow's row. A repeated call of the same workflow (recovery) finds its
    own row and reserves nothing more.

    Args:
        workflow_id: DBOS id of the workflow (one reservation per workflow).
        slug: City slug, for the record.

    Returns:
        ``False`` when the daily budget is spent.
    """
    now = datetime.now(UTC)
    async with transaction() as connection:
        await connection.execute(select(func.pg_advisory_xact_lock(LOCK_KEY)))
        own = (await connection.execute(build_attempt_select(workflow_id))).scalar_one()
        if own:
            return True
        used = (
            await connection.execute(
                build_recent_attempts_select(now - timedelta(days=1))
            )
        ).scalar_one()
        if used >= get_settings().osm.overpass_daily_limit:
            return False
        await connection.execute(build_attempt_insert(workflow_id, slug, now))
    return True


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
    url = f"{get_settings().osm.nominatim_url}/search"
    response = await _send(lambda client: client.get(url, params=params))
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
    url = get_settings().osm.overpass_url
    data = {"data": overpass_query(relation_id)}
    response = await _send(lambda client: client.post(url, data=data))
    body = response.json()
    remark = str(body.get("remark", ""))
    if "runtime error" in remark:
        raise OsmBusyError(remark)
    elements = body.get("elements")
    if not isinstance(elements, list):
        msg = "Overpass answered without an elements list"
        raise TypeError(msg)
    week = monday_of(datetime.now(UTC), timezone)
    mapped = (to_place_row(element, timezone, week) for element in elements)
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
    statement = build_fetch_upsert(slug, relation_id, stored, datetime.now(UTC))
    async with transaction() as connection:
        await connection.execute(statement)
