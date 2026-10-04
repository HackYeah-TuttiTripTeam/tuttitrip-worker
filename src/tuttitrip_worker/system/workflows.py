"""System workflows: ``ping`` (backend smoke test) and the heartbeat schedule."""

from datetime import UTC, datetime
from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import PingInput, PingOutput, Workflow, parse_input
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.dbos.runtime import PORTABLE
from tuttitrip_worker.system import steps


@DBOS.workflow(name=Workflow.PING.value, serialization_type=PORTABLE)
def ping(payload: dict[str, Any]) -> dict[str, Any]:
    """Echo the message with the worker's environment and version.

    Args:
        payload: JSON object matching ``PingInput``.

    Returns:
        JSON object matching ``PingOutput``.
    """
    request = parse_input(PingInput, payload)
    settings = get_settings()
    output = PingOutput(
        message=request.message, worker_app_version=settings.application_version
    )
    return output.model_dump(mode="json")


@DBOS.workflow(name="heartbeat")
async def heartbeat(scheduled_at: datetime, context: object) -> None:
    """Scheduled every 30 s (``SCHEDULED_WORKFLOWS``): refresh the heartbeat row.

    Args:
        scheduled_at: When the schedule fired (unused).
        context: Schedule context (unused).
    """
    del scheduled_at, context
    await steps.upsert_heartbeat()


def beat_now() -> None:
    """Start the heartbeat workflow once, right after launch (not awaited).

    The cron only fires every 30 s, so without this a fresh worker looks
    ``missing`` in the backend's ``/health`` until the first tick. A failure
    shows up as an ``ERROR`` heartbeat workflow; the schedule keeps trying.
    """
    DBOS.start_workflow(heartbeat, datetime.now(UTC), None)
