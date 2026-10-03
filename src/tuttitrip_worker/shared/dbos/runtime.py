"""DBOS configuration, queues and the helpers workflows share.

Order matters (see AGENTS.md, "DBOS rules"):

1. Import every module that defines workflows, steps and agents.
2. :func:`init_dbos` creates the DBOS singleton from settings.
3. ``DBOS.launch()`` starts recovery and the queue runners.
4. :func:`register_queues` persists queue limits and drops queues that left
   the contract (after launch, sync context).
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, TypedDict

from dbos import DBOS, DBOSConfig, WorkflowSerializationFormat

from tuttitrip_worker.contracts import APPLICATION_NAME, PROGRESS_EVENT, Progress, Queue
from tuttitrip_worker.shared.config.settings import Settings

logger = logging.getLogger(APPLICATION_NAME)

PORTABLE: Final = WorkflowSerializationFormat.PORTABLE
"""Serialization for every backend-facing workflow (JSON, not pickle)."""


class QueueRateLimit(TypedDict):
    """At most ``limit`` workflow starts per ``period`` seconds (DBOS limiter)."""

    limit: int
    period: float


@dataclass(frozen=True, slots=True)
class QueueLimits:
    """Flow control of one queue on this worker."""

    worker_concurrency: int | None = None
    limiter: QueueRateLimit | None = None


QUEUE_LIMITS: Final[Mapping[Queue, QueueLimits]] = {
    Queue.DEFAULT: QueueLimits(worker_concurrency=8),
    # One GPU on dellpromaxgb10: SGLang batches internally, so allow a little
    # parallelism but never flood it.
    Queue.LOCAL_LLM: QueueLimits(worker_concurrency=2),
    # OpenRouter: at most 30 workflow starts per minute (global across workers).
    Queue.OPENROUTER: QueueLimits(
        worker_concurrency=8, limiter={"limit": 30, "period": 60}
    ),
}


def build_config(settings: Settings) -> DBOSConfig:
    """DBOS configuration for this worker.

    Args:
        settings: Worker settings.

    Returns:
        The config passed to ``DBOS(config=...)``.
    """
    return {
        "name": APPLICATION_NAME,
        "application_version": settings.application_version,
        "system_database_url": settings.system_database_url(),
        # The backend runs `dbos migrate -r tuttitrip_worker`; the worker
        # role cannot (and must not) run DDL.
        "run_migrations": False,
        "log_level": settings.dbos.log_level,
    }


def init_dbos(settings: Settings) -> None:
    """Create the DBOS singleton (call after importing all workflow modules).

    Args:
        settings: Worker settings.
    """
    DBOS(config=build_config(settings))


def register_queues() -> None:
    """Register (or update) every contract queue and drop stale ones.

    DBOS keeps queues in the system database, so a queue removed from the
    contract would still be polled until its row is deleted. A stale queue
    that still holds queued work is kept (with a warning) instead of
    orphaning those workflows.
    """
    for queue in Queue:
        limits = QUEUE_LIMITS[queue]
        DBOS.register_queue(
            queue.value,
            worker_concurrency=limits.worker_concurrency,
            limiter=limits.limiter,
            on_conflict="always_update",
        )
    contract_queues = {queue.value for queue in Queue}
    for stale in DBOS.list_queues():
        if stale.name in contract_queues or stale.application_name != APPLICATION_NAME:
            continue
        if DBOS.list_queued_workflows(
            queue_name=stale.name, limit=1, load_input=False, load_output=False
        ):
            logger.warning(
                "keeping stale queue %s: it still has queued work", stale.name
            )
            continue
        DBOS.delete_queue(stale.name)
        logger.info("deleted stale queue %s", stale.name)


async def report_progress(stage: str, percent: int) -> None:
    """Publish the ``progress`` event of the current workflow.

    Must be called from a workflow body (DBOS records it as a step).

    Args:
        stage: Short machine-readable stage name.
        percent: Progress from 0 to 100.
    """
    event = Progress(stage=stage, percent=percent)
    await DBOS.set_event_async(
        PROGRESS_EVENT,
        event.model_dump(mode="json"),
        serialization_type=PORTABLE,
    )
