"""I/O steps of the linter domain (database reads)."""

from typing import Any, Final
from uuid import UUID

from dbos import DBOS
from sqlalchemy import Select, select

from tuttitrip_worker.linter.logic.candidates import rank
from tuttitrip_worker.linter.schemas import CatalogPlace
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import pasted_documents, places

PLAN_KIND: Final = "plan"


def build_document_select(document_id: UUID, trip_id: UUID) -> Select[Any]:
    """``SELECT text`` of a pasted plan of one trip.

    Matching on ``trip_id`` too means a job for one trip cannot read another
    trip's document by guessing its id.

    Args:
        document_id: Row id from the job payload.
        trip_id: Trip from the job payload.

    Returns:
        The statement (not executed).
    """
    return select(pasted_documents.c.text).where(
        pasted_documents.c.id == document_id,
        pasted_documents.c.trip_id == trip_id,
        pasted_documents.c.kind == PLAN_KIND,
    )


@DBOS.step(retries_allowed=True, max_attempts=3)
async def load_pasted_plan(document_id: str, trip_id: str) -> str | None:
    """Read the pasted plan text (checkpointed, so a resume does not re-read it).

    Args:
        document_id: ``pasted_documents.id``.
        trip_id: Trip the document must belong to.

    Returns:
        The text, or ``None`` when no such plan document exists.
    """
    statement = build_document_select(UUID(document_id), UUID(trip_id))
    async with transaction() as connection:
        text = (await connection.execute(statement)).scalar_one_or_none()
    return None if text is None else str(text)


MAX_CITY_PLACES: Final = 20000
"""Most places of one city read for matching (a safety cap, Berlin has fewer)."""


def build_city_places_select(city_slug: str) -> Select[Any, Any, Any]:
    """``SELECT id, name, category`` of the catalog places of one city.

    Args:
        city_slug: City slug from the payload.

    Returns:
        The statement (not executed), ordered by id so the cap is stable.
    """
    return (
        select(places.c.id, places.c.name, places.c.category)
        .where(places.c.city_slug == city_slug)
        .order_by(places.c.id)
        .limit(MAX_CITY_PLACES)
    )


@DBOS.step(retries_allowed=True, max_attempts=3)
async def rank_candidates(city_slug: str, names: list[str]) -> dict[str, list[Any]]:
    """Rank the city's catalog places for each pasted name.

    Reads the places and ranks inside one step, so the checkpoint holds only a
    few candidates per name, not the whole catalog of the city.

    Args:
        city_slug: City slug from the payload.
        names: Distinct place names of the parsed items.

    Returns:
        ``RankedPlace`` lists as JSON, per name (best first, at most nine).
    """
    async with transaction() as connection:
        rows = (
            (await connection.execute(build_city_places_select(city_slug)))
            .mappings()
            .all()
        )
    catalog_places = [
        CatalogPlace(str(row["id"]), str(row["name"]), str(row["category"]))
        for row in rows
    ]
    return {
        name: [p.model_dump(mode="json") for p in rank(name, catalog_places)]
        for name in names
    }
