"""Accommodation domain: evidence from pasted offers workflows."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    ExtractOfferEvidenceInput,
    Workflow,
    not_implemented,
    parse_input,
)
from tuttitrip_worker.shared.dbos.runtime import PORTABLE


@DBOS.workflow(name=Workflow.EXTRACT_OFFER_EVIDENCE.value, serialization_type=PORTABLE)
def extract_offer_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    """Stub of ``extract_offer_evidence``; replaced by tuttitrip-worker#25.

    Args:
        payload: JSON object matching ``ExtractOfferEvidenceInput``.

    Raises:
        ContractError: Always, with code ``not_implemented``.
    """
    parse_input(ExtractOfferEvidenceInput, payload)
    raise not_implemented(Workflow.EXTRACT_OFFER_EVIDENCE)
