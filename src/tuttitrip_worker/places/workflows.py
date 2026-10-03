"""Places domain: candidate places from open data workflows."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    FetchPlaceCandidatesInput,
    Workflow,
    not_implemented,
    parse_input,
)
from tuttitrip_worker.shared.dbos.runtime import PORTABLE


@DBOS.workflow(name=Workflow.FETCH_PLACE_CANDIDATES.value, serialization_type=PORTABLE)
def fetch_place_candidates(payload: dict[str, Any]) -> dict[str, Any]:
    """Stub of ``fetch_place_candidates``; replaced by tuttitrip-worker#26.

    Args:
        payload: JSON object matching ``FetchPlaceCandidatesInput``.

    Raises:
        ContractError: Always, with code ``not_implemented``.
    """
    parse_input(FetchPlaceCandidatesInput, payload)
    raise not_implemented(Workflow.FETCH_PLACE_CANDIDATES)
