"""Places domain: candidate places from open data workflows."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    ErrorCode,
    FetchPlaceCandidatesInput,
    FetchPlaceCandidatesOutput,
    Workflow,
    contract_failure,
    parse_input,
)
from tuttitrip_worker.places import steps
from tuttitrip_worker.places.logic.slug import slugify
from tuttitrip_worker.places.schemas import GeocodedCity, RefreshState
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress

PAUSE_BETWEEN_QUERIES_SEC = 1
"""Nominatim allows one request a second; the pause also spaces Overpass calls."""


@DBOS.workflow(name=Workflow.FETCH_PLACE_CANDIDATES.value, serialization_type=PORTABLE)
async def fetch_place_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    """Fetch the places of a city from OpenStreetMap into the catalog.

    Geocodes the city in Nominatim (only when no earlier fetch knows its OSM
    relation), then asks Overpass once for the city's places and upserts them
    as unverified rows with source ``osm``. A city fetched in the last
    ``refresh_days`` days is skipped without any request.

    Args:
        payload: JSON object matching ``FetchPlaceCandidatesInput``.

    Returns:
        ``FetchPlaceCandidatesOutput`` as JSON.

    Raises:
        ContractError: ``city_not_found`` when Nominatim does not know the city
            (or ``city_slug`` names a city nobody fetched yet);
            ``rate_limited`` when today's Overpass budget is spent.
    """
    request = parse_input(FetchPlaceCandidatesInput, payload)
    slug = request.city_slug or slugify(request.city_query or "")
    if not slug:
        raise contract_failure(ErrorCode.CITY_NOT_FOUND, "city name has no letters")
    state = RefreshState.model_validate(await steps.load_refresh_state(slug))
    if state.fresh:
        return FetchPlaceCandidatesOutput(
            city_slug=slug, refreshed=False, stored=0
        ).model_dump(mode="json")
    if state.queries_last_day >= get_settings().osm.overpass_daily_limit:
        raise contract_failure(
            ErrorCode.RATE_LIMITED, "daily Overpass budget spent, try tomorrow"
        )

    await report_progress("locating", 10)
    relation_id, timezone = state.relation_id, state.timezone
    if relation_id is None or timezone is None:
        query = request.city_query or state.city_name
        if query is None:
            raise contract_failure(
                ErrorCode.CITY_NOT_FOUND,
                f"city {slug} is not in the catalog yet, pass city_query",
            )
        found = await steps.geocode_city(query, state.country)
        if found is None:
            raise contract_failure(
                ErrorCode.CITY_NOT_FOUND, f"OpenStreetMap does not know {query!r}"
            )
        city = GeocodedCity.model_validate(found)
        await steps.store_city(slug, found)
        relation_id, timezone = city.relation_id, state.timezone or city.timezone
        await DBOS.sleep_async(PAUSE_BETWEEN_QUERIES_SEC)

    await report_progress("fetching", 40)
    stored = await steps.import_places(slug, relation_id, timezone)
    await steps.mark_fetched(slug, relation_id, stored)
    await report_progress("done", 100)
    return FetchPlaceCandidatesOutput(
        city_slug=slug, refreshed=True, stored=stored
    ).model_dump(mode="json")
