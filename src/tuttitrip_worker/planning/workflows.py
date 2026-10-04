"""Planning workflows: durable LLM drafts of trip plans."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    GenerateTripPlanInput,
    GenerateTripPlanOutput,
    LlmProvider,
    Workflow,
    WriteJustificationsInput,
    not_implemented,
    parse_input,
)
from tuttitrip_worker.planning.agents import planner_agent
from tuttitrip_worker.planning.constants import PROGRESS_DRAFTING, PROGRESS_SAVING
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.dbos.constants import PROGRESS_DONE
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
    await report_progress(*PROGRESS_DRAFTING)
    # Every model request inside this call is a DBOS step (DBOSDurability).
    result = await planner_agent.run(
        f"Trip {request.trip_id}: {request.request}",
        model=model_id(LlmProvider(request.provider)),
    )
    await report_progress(*PROGRESS_SAVING)
    draft = result.output
    output = GenerateTripPlanOutput(
        destination=draft.destination, days=draft.days, highlights=draft.highlights
    ).model_dump(mode="json")
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(
            workflow_id, Workflow.GENERATE_TRIP_PLAN.value, output
        )
    await report_progress(*PROGRESS_DONE)
    return output


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
