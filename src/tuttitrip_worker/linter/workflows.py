"""Linter domain: checking plans pasted from other tools workflows."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    SUPPORTED_CONTRACT_VERSIONS,
    ContractError,
    ContractErrorData,
    ErrorCode,
    ParsePastedPlanInput,
    ParsePastedPlanOutput,
    Workflow,
    parse_input,
)
from tuttitrip_worker.linter import steps
from tuttitrip_worker.linter.agents import parser_agent
from tuttitrip_worker.linter.logic.quotes import split_items
from tuttitrip_worker.linter.schemas import PastedText
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress
from tuttitrip_worker.shared.llm.models import ModelKey, model_id


def _document_not_found(document_id: str) -> ContractError:
    data = ContractErrorData(
        code=ErrorCode.INVALID_PAYLOAD,
        supported_versions=sorted(SUPPORTED_CONTRACT_VERSIONS),
        errors=[
            {
                "loc": ["document_id"],
                "msg": "no pasted plan with this id for this trip",
                "input": document_id,
            }
        ],
    )
    return ContractError(f"pasted plan {document_id} not found for this trip", data)


@DBOS.workflow(name=Workflow.PARSE_PASTED_PLAN.value, serialization_type=PORTABLE)
async def parse_pasted_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Turn a pasted plan into items that each carry a verbatim quote.

    Reads the text by document id, lets the parser agent extract items (every
    model request is a DBOS step), keeps only items whose quote occurs in the
    text and stores the result in ``job_results``.

    Args:
        payload: JSON object matching ``ParsePastedPlanInput``.

    Returns:
        JSON object matching ``ParsePastedPlanOutput``.

    Raises:
        ContractError: The document does not exist for this trip.
    """
    request = parse_input(ParsePastedPlanInput, payload)
    await report_progress("loading", 5)
    text = await steps.load_pasted_plan(str(request.document_id), str(request.trip_id))
    if text is None:
        raise _document_not_found(str(request.document_id))
    await report_progress("reading", 15)
    result = await parser_agent.run(
        f"City: {request.city_slug}\n<pasted_text>\n{text}\n</pasted_text>",
        deps=PastedText(text),
        # The queue follows `provider`; the chat model is Qwen on the GB10 with
        # OpenRouter as the fallback inside the catalog.
        model=model_id(ModelKey.CHAT),
    )
    items, unread = split_items(result.output.items, text)
    # The catalog matching step (tuttitrip-worker#24) fills `matches` here.
    output = ParsePastedPlanOutput(items=items, unread=unread).model_dump(mode="json")
    await report_progress("saving", 90)
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(
            workflow_id, Workflow.PARSE_PASTED_PLAN.value, output
        )
    await report_progress("done", 100)
    return output
