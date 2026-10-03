"""Persistent job results (``job_results``) for outputs worth keeping.

Small results travel as the workflow output; this table is for large or
long-lived ones the backend reads by ``workflow_id``.
"""

from typing import Any

from dbos import DBOS
from sqlalchemy.dialects.postgresql import Insert, insert

from tuttitrip_worker.contracts import CONTRACT_VERSION
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import job_results


def build_job_result_upsert(
    workflow_id: str, workflow_name: str, result: dict[str, Any]
) -> Insert:
    """Build the idempotent upsert of one ``job_results`` row.

    Args:
        workflow_id: Row key.
        workflow_name: Contract workflow name.
        result: JSON object to store.

    Returns:
        The statement (not executed).
    """
    statement = insert(job_results).values(
        workflow_id=workflow_id,
        workflow_name=workflow_name,
        contract_version=CONTRACT_VERSION,
        result=result,
    )
    return statement.on_conflict_do_update(
        index_elements=[job_results.c.workflow_id],
        set_={
            "workflow_name": statement.excluded.workflow_name,
            "contract_version": statement.excluded.contract_version,
            "result": statement.excluded.result,
        },
    )


@DBOS.step(retries_allowed=True, max_attempts=3)
async def save_job_result(
    workflow_id: str, workflow_name: str, result: dict[str, Any]
) -> None:
    """Upsert the result of a workflow (idempotent per ``workflow_id``).

    Args:
        workflow_id: DBOS workflow id (deterministic, chosen by the backend).
        workflow_name: Contract workflow name.
        result: JSON object to store.
    """
    statement = build_job_result_upsert(workflow_id, workflow_name, result)
    async with transaction() as connection:
        await connection.execute(statement)
