"""Scheduled workflow: delete old notifications once a day."""

import logging
from datetime import datetime

from dbos import DBOS

from tuttitrip_worker.contracts import APPLICATION_NAME
from tuttitrip_worker.notifications import steps
from tuttitrip_worker.notifications.logic.retention import (
    BATCH_SIZE,
    MAX_BATCHES,
    cutoffs,
)
from tuttitrip_worker.notifications.schemas import PurgeResult

logger = logging.getLogger(APPLICATION_NAME)


@DBOS.workflow(name="purge_notifications")
async def purge_notifications(
    scheduled_at: datetime, context: object
) -> dict[str, int]:
    """Scheduled daily (``SCHEDULED_WORKFLOWS``): delete old notifications.

    Read ones older than 90 days and all older than 180 days go, in batches of
    1000 rows (one short transaction each), and the total is logged.

    Args:
        scheduled_at: When the schedule fired; the cutoffs are computed from it,
            so a recovered run uses the same ones.
        context: Schedule context (unused).

    Returns:
        How many rows were deleted in how many batches.
    """
    del context
    limits = cutoffs(scheduled_at)
    deleted = batches = 0
    while batches < MAX_BATCHES:
        count = await steps.purge_batch(limits.read, limits.any)
        batches += 1
        deleted += count
        if count < BATCH_SIZE:
            break
    logger.info("purged %d notifications in %d batches", deleted, batches)
    result = PurgeResult(deleted=deleted, batches=batches)
    return {"deleted": result.deleted, "batches": result.batches}
