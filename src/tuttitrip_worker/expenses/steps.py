"""I/O steps of the expenses domain: the evidence image and the trip dates.

The image is read and sent to the vision model inside ONE step, and the step
returns only the structured reading. DBOS stores step outputs, so bytes of the
image never reach the system database; a nested durable agent inside a step
runs as plain code, so its requests are not checkpointed either.
"""

from typing import Any
from uuid import UUID

from dbos import DBOS
from sqlalchemy import Select, select

from tuttitrip_worker.expenses.agents import read_receipt_image
from tuttitrip_worker.expenses.schemas import ReceiptReading
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import expense_evidence, trips


def build_evidence_select(evidence_id: UUID, trip_id: UUID) -> Select[Any, Any]:
    """``SELECT data, media_type`` of one image of one trip.

    Matching on ``trip_id`` too means a job for one trip cannot read another
    trip's image by guessing its id.

    Args:
        evidence_id: Row id from the job payload.
        trip_id: Trip from the job payload.

    Returns:
        The statement (not executed).
    """
    return select(expense_evidence.c.data, expense_evidence.c.media_type).where(
        expense_evidence.c.id == evidence_id, expense_evidence.c.trip_id == trip_id
    )


def build_trip_dates_select(
    trip_id: UUID,
) -> Select[Any, Any]:
    """``SELECT start_date, end_date`` of a trip.

    Args:
        trip_id: Trip from the job payload.

    Returns:
        The statement (not executed).
    """
    return select(trips.c.start_date, trips.c.end_date).where(trips.c.id == trip_id)


@DBOS.step(retries_allowed=True, max_attempts=2)
async def read_evidence(evidence_id: str, trip_id: str) -> dict[str, Any] | None:
    """Load the image and let the vision model read it; return the reading only.

    Args:
        evidence_id: ``expense_evidence.id``.
        trip_id: Trip the image must belong to.

    Returns:
        A ``ReceiptReading`` as JSON, or ``None`` when no such image exists.
    """
    statement = build_evidence_select(UUID(evidence_id), UUID(trip_id))
    async with transaction() as connection:
        row = (await connection.execute(statement)).mappings().first()
    if row is None:
        return None
    reading = await read_receipt_image(bytes(row["data"]), str(row["media_type"]))
    return reading.model_dump(mode="json")


@DBOS.step(retries_allowed=True, max_attempts=3)
async def load_trip_dates(trip_id: str) -> dict[str, str | None]:
    """Read when the trip runs.

    Args:
        trip_id: Trip from the payload.

    Returns:
        ``{"start": ISO date or None, "end": ...}``; both ``None`` for an unknown trip.
    """
    async with transaction() as connection:
        row = (
            (await connection.execute(build_trip_dates_select(UUID(trip_id))))
            .mappings()
            .first()
        )
    start, end = (row["start_date"], row["end_date"]) if row else (None, None)
    return {
        "start": str(start) if start else None,  # str(date) is ISO 8601
        "end": str(end) if end else None,
    }


def to_reading(data: dict[str, Any]) -> ReceiptReading:
    """Rebuild a reading from the step output.

    Args:
        data: JSON from :func:`read_evidence`.

    Returns:
        The reading.
    """
    return ReceiptReading.model_validate(data)
