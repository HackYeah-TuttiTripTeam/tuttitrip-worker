"""I/O steps of the notifications domain: the daily purge, in small batches."""

from datetime import datetime

from dbos import DBOS
from sqlalchemy import Delete, and_, delete, or_, select

from tuttitrip_worker.notifications.logic.retention import BATCH_SIZE
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import notifications


def build_purge_batch(read_before: datetime, any_before: datetime) -> Delete:
    """``DELETE`` of up to one batch of old rows.

    Old means created before ``read_before`` and already read, or created before
    ``any_before``. The subselect limits the batch (``DELETE`` has no ``LIMIT``
    in PostgreSQL). Deleting sends no ``NOTIFY``: the trigger is on INSERT only.

    Args:
        read_before: Read rows created before this are deleted.
        any_before: Every row created before this is deleted.

    Returns:
        The statement (not executed).
    """
    old = select(notifications.c.id).where(
        or_(
            and_(
                notifications.c.read_at.is_not(None),
                notifications.c.created_at < read_before,
            ),
            notifications.c.created_at < any_before,
        )
    )
    return delete(notifications).where(
        notifications.c.id.in_(old.limit(BATCH_SIZE).scalar_subquery())
    )


@DBOS.step(retries_allowed=True, max_attempts=3)
async def purge_batch(read_before: datetime, any_before: datetime) -> int:
    """Delete one batch of old notifications in its own short transaction.

    Args:
        read_before: Read rows created before this are deleted.
        any_before: Every row created before this is deleted.

    Returns:
        How many rows were deleted (fewer than a batch means nothing is left).
    """
    async with transaction() as connection:
        result = await connection.execute(build_purge_batch(read_before, any_before))
    return result.rowcount
