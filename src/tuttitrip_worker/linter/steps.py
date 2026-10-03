"""I/O steps of the linter domain (database reads)."""

from typing import Any, Final
from uuid import UUID

from dbos import DBOS
from sqlalchemy import Select, select

from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import pasted_documents

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
