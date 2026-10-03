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
async def load_offer_text(document_id: UUID, trip_id: UUID) -> str | None:
    """Read the pasted offer (the worker has SELECT on ``pasted_documents`` only).

    Args:
        document_id: Id of the pasted document.
        trip_id: Trip it must belong to.

    Returns:
        The offer text, or ``None`` when there is no such offer for the trip.
    """
    async with transaction() as connection:
        result = await connection.execute(offer_query(document_id, trip_id))
        text = result.scalar_one_or_none()
        return None if text is None else str(text)
