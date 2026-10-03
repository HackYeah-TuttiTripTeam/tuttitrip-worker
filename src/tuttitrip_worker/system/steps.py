"""I/O steps of the system domain."""

from datetime import UTC, datetime

from dbos import DBOS
from sqlalchemy.dialects.postgresql import insert

from tuttitrip_worker.contracts import APPLICATION_NAME, SUPPORTED_CONTRACT_VERSIONS
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import worker_heartbeats
from tuttitrip_worker.system.schemas import Heartbeat


def current_heartbeat() -> Heartbeat:
    """Describe this worker as of now.

    Returns:
        The heartbeat row to upsert.
    """
    settings = get_settings()
    return Heartbeat(
        worker_id=f"{APPLICATION_NAME}-{settings.environment}",
        env=settings.environment,
        contract_version=max(SUPPORTED_CONTRACT_VERSIONS),
        min_contract_version=min(SUPPORTED_CONTRACT_VERSIONS),
        app_version=settings.application_version,
        last_seen=datetime.now(UTC),
    )


@DBOS.step(retries_allowed=True, max_attempts=3)
async def upsert_heartbeat() -> None:
    """Insert or refresh this worker's row in ``worker_heartbeats``."""
    row = current_heartbeat().model_dump()
    statement = insert(worker_heartbeats).values(**row)
    statement = statement.on_conflict_do_update(
        index_elements=[worker_heartbeats.c.worker_id],
        set_={key: statement.excluded[key] for key in row if key != "worker_id"},
    )
    async with transaction() as connection:
        await connection.execute(statement)
