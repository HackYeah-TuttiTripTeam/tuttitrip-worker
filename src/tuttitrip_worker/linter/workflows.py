"""Linter domain: checking plans pasted from other tools workflows."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    ParsePastedPlanInput,
    ParsePastedPlanOutput,
    Workflow,
    document_not_found,
    parse_input,
)
from tuttitrip_worker.linter import steps
from tuttitrip_worker.linter.agents import read_pasted_plan
from tuttitrip_worker.linter.constants import (
    PROGRESS_LOADING,
    PROGRESS_MATCHING,
    PROGRESS_READING,
    PROGRESS_SAVING,
)
from tuttitrip_worker.linter.logic.quotes import split_items
from tuttitrip_worker.linter.services.match import match_items
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.dbos.constants import PROGRESS_DONE
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress


@DBOS.workflow(name=Workflow.PARSE_PASTED_PLAN.value, serialization_type=PORTABLE)
async def parse_pasted_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn a pasted plan into items that each carry a verbatim quote.

    Reads the text by document id, lets the parser agent extract items (every
    model request is a DBOS step), keeps only items the text backs up, matches
    them to the city's catalog places and stores the result in ``job_results``.
    The queue follows ``provider``; the model is always the catalog's chat
    model (Qwen on the GB10, OpenRouter as fallback), which is the agent's
    default.

    Args:
        payload: JSON object matching ``ParsePastedPlanInput``.

    Returns:
        JSON object matching ``ParsePastedPlanOutput``.

    Raises:
        ContractError: The plan document does not exist for this trip
            (``document_not_found``) or the model never produced a valid
            structured answer (``model_output_invalid``).
    """
    request = parse_input(ParsePastedPlanInput, payload)
    await report_progress(*PROGRESS_LOADING)
    text = await steps.load_pasted_plan(str(request.document_id), str(request.trip_id))
    if text is None:
        raise document_not_found(request.document_id, steps.PLAN_KIND)
    await report_progress(*PROGRESS_READING)
    draft = await read_pasted_plan(text, str(request.document_id), request.city_slug)
    items, unread = split_items(draft.items, text)
    await report_progress(*PROGRESS_MATCHING)
    matches = await match_items(items, request.city_slug, str(request.document_id))
    output = ParsePastedPlanOutput(
        items=items, unread=unread, matches=matches
    ).model_dump(mode="json")
    await report_progress(*PROGRESS_SAVING)
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(
            workflow_id, Workflow.PARSE_PASTED_PLAN.value, output
        )
    await report_progress(*PROGRESS_DONE)
    return output
