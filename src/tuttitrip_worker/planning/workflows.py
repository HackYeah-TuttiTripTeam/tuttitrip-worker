"""Planning workflows: durable LLM drafts of trip plans."""

import logging
from typing import Any
from uuid import UUID

from dbos import DBOS

from tuttitrip_worker.contracts import (
    APPLICATION_NAME,
    GenerateTripPlanInput,
    GenerateTripPlanOutput,
    Justification,
    LlmProvider,
    NotificationDraft,
    Workflow,
    WriteJustificationsInput,
    WriteJustificationsOutput,
    document_not_found,
    parse_input,
)
from tuttitrip_worker.planning import steps
from tuttitrip_worker.planning.agents import planner_agent, write_batch
from tuttitrip_worker.planning.constants import (
    PLAN_VERSION_KIND,
    PROGRESS_DRAFTING,
    PROGRESS_LOADING,
    PROGRESS_SAVING,
    PROGRESS_WRITING_FROM,
    PROGRESS_WRITING_SPAN,
    PROGRESS_WRITING_STAGE,
)
from tuttitrip_worker.planning.logic.verdict_facts import split_batches
from tuttitrip_worker.planning.schemas import VerdictBatch, VerdictFacts
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.db.notifications import notify_user
from tuttitrip_worker.shared.dbos.constants import PROGRESS_DONE
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress
from tuttitrip_worker.shared.llm.models import model_id

logger = logging.getLogger(APPLICATION_NAME)


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
    await _notify_plan_ready(request.trip_id, output["destination"], workflow_id)
    await report_progress(*PROGRESS_DONE)
    return output


async def _notify_plan_ready(
    trip_id: UUID, destination: str, workflow_id: str | None
) -> None:
    """Tell the user who started the job that the plan is ready.

    A failure (after the step's retries) is logged and swallowed: the plan is
    already saved, so a notification must not turn a finished job into an error.

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
    try:
        await notify_user(user, draft.model_dump(mode="json"))
    except Exception:
        logger.exception("plan_ready notification for trip %s was not written", trip_id)


@DBOS.workflow(name=Workflow.WRITE_JUSTIFICATIONS.value, serialization_type=PORTABLE)
async def write_justifications(payload: dict[str, Any]) -> dict[str, Any]:
    """Word the verdicts of a plan version, from the algorithm's numbers only.

    Reads the verdicts and ``explain()`` of the stored plan version, lets the
    justifier agent write at most two sentences per place in batches (every
    model request is a DBOS step), keeps only the texts whose numbers and names
    are all in the data and stores the result in ``job_results``. A place
    without a text has none in the output; the backend uses its template. The
    backend enqueues with a deterministic workflow id (``return-existing``), so
    a repeated request costs no second model call.

    Args:
        payload: JSON object matching ``WriteJustificationsInput``.

    Returns:
        JSON object matching ``WriteJustificationsOutput``.

    Raises:
        ContractError: The plan version does not exist (``document_not_found``).
    """
    request = parse_input(WriteJustificationsInput, payload)
    await report_progress(*PROGRESS_LOADING)
    rows = await steps.load_verdict_facts(str(request.plan_id))
    if rows is None:
        raise document_not_found(request.plan_id, PLAN_VERSION_KIND)
    facts = [VerdictFacts.model_validate(row) for row in rows]
    batches = split_batches(facts, get_settings().planning.justification_batch_size)
    justifications: list[Justification] = []
    for index, batch in enumerate(batches):
        percent = PROGRESS_WRITING_FROM + PROGRESS_WRITING_SPAN * index // len(batches)
        await report_progress(PROGRESS_WRITING_STAGE, percent)
        items = await write_batch(VerdictBatch(facts=batch), request.locale)
        justifications += [
            Justification(place_id=item.place_id, text=item.text, source="model")
            for item in items
        ]
    output = WriteJustificationsOutput(justifications=justifications).model_dump(
        mode="json"
    )
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(
            workflow_id, Workflow.WRITE_JUSTIFICATIONS.value, output
        )
    await report_progress(*PROGRESS_DONE)
    return output
