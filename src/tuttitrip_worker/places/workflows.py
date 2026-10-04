"""Places domain: candidate places from open data workflows."""

import asyncio
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
from tuttitrip_worker.places.constants import (
    PROGRESS_ENRICHING,
    PROGRESS_FETCHING,
    PROGRESS_LOCATING,
)
from tuttitrip_worker.places.logic.slug import slugify
from tuttitrip_worker.places.schemas import GeocodedCity, RefreshState
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.dbos.constants import PROGRESS_DONE
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress

PAUSE_BETWEEN_QUERIES_SEC = 1
"""Durable pause between the geocoding and the Overpass query. The spacing
across workflows is enforced by the gate in ``steps``."""


async def _enrich(slug: str) -> int:
    """Research the city's top places on the web within the cost limit.

    The places are researched in chunks of ``concurrency`` (one step each, so
    the step order is fixed); after every chunk the spent cost is added up and
    the loop stops at ``max_cost_usd``. Failures are skipped, not raised.

    Args:
        slug: City slug.

    Returns:
        How many places were stored.
    """
    settings = get_settings().enrich
    targets = await steps.select_research_targets(slug)
    if not targets:
        return 0
    await report_progress(*PROGRESS_ENRICHING)
    spent, stored = 0.0, 0
    for start in range(0, len(targets), settings.concurrency):
        chunk = targets[start : start + settings.concurrency]
        results = await asyncio.gather(*(steps.research_place(t) for t in chunk))
        stored += await steps.store_research(list(results))
        spent += sum(float(result["cost_usd"]) for result in results)
        if spent >= settings.max_cost_usd:
            break
    return stored


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
        enriched = await _enrich(slug)
        return FetchPlaceCandidatesOutput(
            city_slug=slug, refreshed=False, stored=0, enriched=enriched
        ).model_dump(mode="json")

    await report_progress(*PROGRESS_LOCATING)
    relation_id, timezone = state.relation_id, state.timezone
    if request.city_query is not None or relation_id is None or timezone is None:
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
        if relation_id is not None and relation_id != city.relation_id:
            raise contract_failure(
                ErrorCode.SLUG_CONFLICT,
                f"{slug} is already OpenStreetMap relation {relation_id}, "
                f"{query!r} is {city.relation_id}; add the country to the name",
            )
        await steps.store_city(slug, found)
        relation_id, timezone = city.relation_id, state.timezone or city.timezone
        await DBOS.sleep_async(PAUSE_BETWEEN_QUERIES_SEC)

    if not await steps.reserve_overpass_slot(DBOS.workflow_id or slug, slug):
        raise contract_failure(
            ErrorCode.RATE_LIMITED, "daily Overpass budget spent, try tomorrow"
        )
    await report_progress(*PROGRESS_FETCHING)
    stored = await steps.import_places(slug, relation_id, timezone)
    await steps.mark_fetched(slug, relation_id, stored)
    enriched = await _enrich(slug)
    await report_progress(*PROGRESS_DONE)
    return FetchPlaceCandidatesOutput(
        city_slug=slug, refreshed=True, stored=stored, enriched=enriched
    ).model_dump(mode="json")
