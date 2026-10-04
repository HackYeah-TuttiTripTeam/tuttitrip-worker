"""I/O steps of the planning domain: reading plan versions."""

from typing import Any
from uuid import UUID

from dbos import DBOS
from sqlalchemy import Select, select

from tuttitrip_worker.planning.logic.verdict_facts import build_facts
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import plan_versions


def build_plan_select(plan_id: UUID) -> Select[Any]:
    """``SELECT result`` of one plan version.

    Args:
        plan_id: ``plan_versions.id`` from the job payload.

    Returns:
        The statement (not executed).
    """
    return select(plan_versions.c.result).where(plan_versions.c.id == plan_id)


@DBOS.step(retries_allowed=True, max_attempts=get_settings().dbos.step_max_attempts)
async def load_verdict_facts(plan_id: str) -> list[dict[str, Any]] | None:
    """Read the verdicts of a plan version and cut out what a text may use.

    Args:
        plan_id: ``plan_versions.id``.

    Returns:
        ``VerdictFacts`` as JSON objects (step outputs are checkpointed), or
        ``None`` when there is no such plan version.
    """
    async with transaction() as connection:
        result = (await connection.execute(build_plan_select(UUID(plan_id)))).scalar()
    if result is None:
        return None
    return [fact.model_dump(mode="json") for fact in build_facts(result)]
