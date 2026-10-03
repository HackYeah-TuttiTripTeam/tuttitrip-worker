"""Accommodation workflows: quotes from a pasted offer and their assessment."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.accommodation import agents, steps
from tuttitrip_worker.accommodation.logic.evidence import (
    Assessment,
    build_evidence,
    verified_quotes,
)
from tuttitrip_worker.contracts import (
    ExtractOfferEvidenceInput,
    ExtractOfferEvidenceOutput,
    Workflow,
    document_not_found,
    parse_input,
)
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress


@DBOS.workflow(name=Workflow.EXTRACT_OFFER_EVIDENCE.value, serialization_type=PORTABLE)
async def extract_offer_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Quote the pasted offer for each requirement and assess every quote.

    The extractor (language model) proposes quotes, pure code keeps only the
    ones that are verbatim in the offer, and the decision model assesses each
    surviving quote. A requirement the offer is silent about has no quotes and
    no assessment. When no model can assess, quotes are returned without
    assessment and the workflow still succeeds.

    Args:
        payload: JSON object matching ``ExtractOfferEvidenceInput``.

    Returns:
        JSON object matching ``ExtractOfferEvidenceOutput``.

    Raises:
        ContractError: The offer does not exist for the trip.
    """
    request = parse_input(ExtractOfferEvidenceInput, payload)
    await report_progress("reading", 5)
    offer = await steps.load_offer_text(request.document_id, request.trip_id)
    if offer is None:
        raise document_not_found(request.document_id, steps.OFFER_KIND)

    keys = request.requirement_keys
    labels = request.requirements or []
    await report_progress("quoting", 15)
    extracted = await agents.extract_quotes(offer, keys, labels)
    quotes = verified_quotes(offer, extracted, keys)

    named = {item.key: item.label for item in labels}
    assessments: dict[tuple[str, str], Assessment] = {}
    pairs = [(key, text) for key in keys for text in quotes[key]]
    for done, (key, text) in enumerate(pairs):
        await report_progress("assessing", 30 + 60 * done // len(pairs))
        judged = await agents.judge_quote(key, named.get(key), text)
        if judged is not None:
            assessments[key, text] = judged

    output = ExtractOfferEvidenceOutput(
        evidence=build_evidence(keys, quotes, assessments)
    ).model_dump(mode="json")
    await report_progress("saving", 95)
    workflow_id = DBOS.workflow_id
    if workflow_id is not None:
        await job_results.save_job_result(
            workflow_id, Workflow.EXTRACT_OFFER_EVIDENCE.value, output
        )
    await report_progress("done", 100)
    return output
