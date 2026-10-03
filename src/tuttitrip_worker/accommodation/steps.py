"""I/O steps of the accommodation domain (database reads)."""

from typing import Any
from uuid import UUID

from dbos import DBOS
from sqlalchemy import Select, select

from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import pasted_documents

OFFER_KIND = "offer"


def offer_query(document_id: UUID, trip_id: UUID) -> Select[Any]:
    """``SELECT text`` of an offer document of one trip.

    Args:
        document_id: Id of the pasted document.
        trip_id: Trip it must belong to.

    Returns:
        The statement (not executed).
    """
    return select(pasted_documents.c.text).where(
        pasted_documents.c.id == document_id,
        pasted_documents.c.trip_id == trip_id,
        pasted_documents.c.kind == OFFER_KIND,
    )


@DBOS.step(retries_allowed=True, max_attempts=3)
async def load_offer_text(document_id: str, trip_id: str) -> str | None:
    """Read the pasted offer (the worker has SELECT on ``pasted_documents`` only).

    Ids are strings because step arguments of a portable workflow are JSON.

    Args:
        document_id: Id of the pasted document (UUID text).
        trip_id: Trip it must belong to (UUID text).

    Returns:
        The offer text, or ``None`` when there is no such offer for the trip.
    """
    query = offer_query(UUID(document_id), UUID(trip_id))
    async with transaction() as connection:
        result = await connection.execute(query)
        text = result.scalar_one_or_none()
        return None if text is None else str(text)
