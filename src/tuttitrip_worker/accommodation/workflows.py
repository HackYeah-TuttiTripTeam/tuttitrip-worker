"""Accommodation workflows: quotes from a pasted offer and their assessment."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.accommodation import steps
from tuttitrip_worker.accommodation.constants import (
    PROGRESS_ASSESSING,
    PROGRESS_READING,
    PROGRESS_SAVING,
)
from tuttitrip_worker.accommodation.services.assess import assess_offer
from tuttitrip_worker.contracts import (
    ExtractOfferEvidenceInput,
    ExtractOfferEvidenceOutput,
    Workflow,
    document_not_found,
    parse_input,
)
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.dbos.constants import PROGRESS_DONE
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress


@DBOS.workflow(name=Workflow.EXTRACT_OFFER_EVIDENCE.value, serialization_type=PORTABLE)
async def extract_offer_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Quote the pasted offer for each requirement and assess every quote.

    See :func:`assess_offer` for the pipeline and the meaning of the result:
    ``quotes == []`` is a silent offer, ``verdict is None`` is an unavailable
    judge (the workflow still succeeds). The workflow itself only reads the
    offer and persists the result.

    Args:
        payload: JSON object matching ``ExtractOfferEvidenceInput``.

    Returns:
        JSON object matching ``ExtractOfferEvidenceOutput``.

    Raises:
        ContractError: The offer does not exist for the trip.
    """
    request = parse_input(ExtractOfferEvidenceInput, payload)
    await report_progress(*PROGRESS_READING)
    offer = await steps.load_offer_text(str(request.document_id), str(request.trip_id))
    if offer is None:
        raise document_not_found(request.document_id, steps.OFFER_KIND)

    await report_progress(*PROGRESS_ASSESSING)
    evidence = await assess_offer(
        offer, request.requirement_keys, request.requirements or []
    )
    output = ExtractOfferEvidenceOutput(evidence=evidence).model_dump(mode="json")
    await report_progress(*PROGRESS_SAVING)
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(
            workflow_id, Workflow.EXTRACT_OFFER_EVIDENCE.value, output
        )
    await report_progress(*PROGRESS_DONE)
    return output
