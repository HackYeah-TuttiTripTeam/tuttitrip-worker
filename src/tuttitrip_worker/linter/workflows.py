"""Linter domain: checking plans pasted from other tools workflows."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    ParsePastedPlanInput,
    Workflow,
    not_implemented,
    parse_input,
)
from tuttitrip_worker.shared.dbos.runtime import PORTABLE


@DBOS.workflow(name=Workflow.PARSE_PASTED_PLAN.value, serialization_type=PORTABLE)
def parse_pasted_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """Stub of ``parse_pasted_plan``; replaced by tuttitrip-worker#23.

    Args:
        payload: JSON object matching ``ParsePastedPlanInput``.

    Raises:
        ContractError: Always, with code ``not_implemented``.
    """
    parse_input(ParsePastedPlanInput, payload)
    raise not_implemented(Workflow.PARSE_PASTED_PLAN)
