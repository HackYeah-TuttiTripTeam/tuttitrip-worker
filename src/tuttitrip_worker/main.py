"""Composition root: register workflows, launch DBOS, serve until SIGTERM.

Importing this module imports every workflow module, so all workflows, steps
and agents are registered before :func:`run` calls ``DBOS.launch()``.
"""

import logging
import signal
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from types import FrameType
from typing import Any, Final

from dbos import DBOS

from tuttitrip_worker.contracts import (
    APPLICATION_NAME,
    CONTRACT_VERSION,
    SCHEDULED_WORKFLOWS,
    Workflow,
)
from tuttitrip_worker.embeddings.workflows import embed_texts
from tuttitrip_worker.healthcheck import LIVENESS_FILE, LIVENESS_INTERVAL_SEC
from tuttitrip_worker.planning.workflows import generate_trip_plan
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.dbos.runtime import (
    all_queues,
    init_dbos,
    register_queues,
)
from tuttitrip_worker.system.workflows import heartbeat, ping

logger = logging.getLogger(APPLICATION_NAME)

WORKFLOWS: Final[Mapping[Workflow, Callable[..., Any]]] = {
    Workflow.PING: ping,
    Workflow.GENERATE_TRIP_PLAN: generate_trip_plan,
    Workflow.EMBED_TEXTS: embed_texts,
}
"""Every contract workflow and the function registered under its name."""

SCHEDULES: Final[Mapping[str, Callable[..., Any]]] = {"heartbeat": heartbeat}
"""Every scheduled workflow (cron in ``contracts.SCHEDULED_WORKFLOWS``)."""


def apply_schedules() -> None:
    """Create or replace the cron schedules (idempotent, after launch)."""
    DBOS.apply_schedules(
        [
            {
                "schedule_name": name,
                "workflow_fn": workflow,
                "schedule": SCHEDULED_WORKFLOWS[name],
            }
            for name, workflow in SCHEDULES.items()
        ]
    )


def touch_liveness(path: Path) -> None:
    """Mark the worker alive if the system database answers.

    Args:
        path: File whose mtime the Docker healthcheck reads.
    """
    try:
        DBOS.list_queued_workflows(limit=1)
    except Exception:
        logger.exception("system database check failed")
        return
    path.touch()


def run() -> None:
    """Launch DBOS and process queues until SIGTERM/SIGINT."""
    settings = get_settings()
    logging.basicConfig(level=settings.dbos.log_level)
    init_dbos(settings)
    DBOS.launch()
    register_queues()
    apply_schedules()
    logger.info(
        "worker ready: env=%s app_version=%s contract=%s queues=%s",
        settings.environment,
        settings.application_version,
        CONTRACT_VERSION,
        ",".join(all_queues()),
    )

    stop = threading.Event()

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        logger.info("received %s, shutting down", signal.Signals(signum).name)
        stop.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    touch_liveness(LIVENESS_FILE)
    while not stop.wait(LIVENESS_INTERVAL_SEC):
        touch_liveness(LIVENESS_FILE)

    # Lets running workflows finish for a while; anything still running stays
    # PENDING and is recovered by the next start (same executor and version).
    DBOS.destroy(workflow_completion_timeout_sec=settings.dbos.shutdown_timeout_sec)
    LIVENESS_FILE.unlink(missing_ok=True)
    logger.info("worker stopped")


def main() -> None:
    """Console-script entry point (``tuttitrip-worker``)."""
    run()
