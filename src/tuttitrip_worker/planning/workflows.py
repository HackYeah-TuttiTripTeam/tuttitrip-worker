"""Planning workflows: durable LLM drafts of trip plans."""

from typing import Any
from uuid import UUID

from dbos import DBOS

from tuttitrip_worker.contracts import (
    GenerateTripPlanInput,
    GenerateTripPlanOutput,
    LlmProvider,
    NotificationDraft,
    Workflow,
    WriteJustificationsInput,
    not_implemented,
    parse_input,
)
from tuttitrip_worker.planning.agents import planner_agent
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.db.notifications import notify_user
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress
from tuttitrip_worker.shared.llm.models import model_id


@DBOS.workflow(name=Workflow.GENERATE_TRIP_PLAN.value, serialization_type=PORTABLE)
async def generate_trip_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Draft a trip plan with the planner agent and keep it in ``job_results``.

    Args:
        payload: JSON object matching ``GenerateTripPlanInput``.

    Returns:
        JSON object matching ``GenerateTripPlanOutput``.
    """
    request = parse_input(GenerateTripPlanInput, payload)
    await report_progress("drafting", 10)
    # Every model request inside this call is a DBOS step (DBOSDurability).
    result = await planner_agent.run(
        f"Trip {request.trip_id}: {request.request}",
        model=model_id(LlmProvider(request.provider)),
    )
    await report_progress("saving", 90)
    draft = result.output
    output = GenerateTripPlanOutput(
        destination=draft.destination, days=draft.days, highlights=draft.highlights
    ).model_dump(mode="json")
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(
            workflow_id, Workflow.GENERATE_TRIP_PLAN.value, output
        )
    await _notify_plan_ready(request.trip_id, output["destination"], workflow_id)
    await report_progress("done", 100)
    return output


async def _notify_plan_ready(
    trip_id: UUID, destination: str, workflow_id: str | None
) -> None:
    """Tell the user who started the job that the plan is ready.

    The recipient is the workflow's authenticated user (the backend sets it to
    the caller's ``sub`` when it enqueues); without one nobody is told. The key
    names the trip and this workflow, so a retried step adds nothing.

    Args:
        trip_id: Trip of the plan.
        destination: Destination of the draft, shown in the notification text.
        workflow_id: Id of this workflow (the version of the plan).
    """
    user = DBOS.authenticated_user
    if user is None or workflow_id is None:
        return
    draft = NotificationDraft(
        type="plan_ready",
        trip_id=trip_id,
        params={"destination": destination[:100]},
        actions=["open_plan"],
        dedupe_key=f"plan_ready:{trip_id}:{workflow_id}",
    )
    await notify_user(user, draft.model_dump(mode="json"))


@DBOS.workflow(name=Workflow.WRITE_JUSTIFICATIONS.value, serialization_type=PORTABLE)
def write_justifications(payload: dict[str, Any]) -> dict[str, Any]:
    """Stub of ``write_justifications``; replaced by tuttitrip-worker#27.

    Args:
        payload: JSON object matching ``WriteJustificationsInput``.

    Raises:
        ContractError: Always, with code ``not_implemented``.
    """
    parse_input(WriteJustificationsInput, payload)
    raise not_implemented(Workflow.WRITE_JUSTIFICATIONS)
