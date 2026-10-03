"""Internal payloads of the system domain (pure: Pydantic only)."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class Heartbeat(BaseModel):
    """One row of ``worker_heartbeats``: proof that this worker is alive."""

    model_config = ConfigDict(frozen=True)

    worker_id: str
    env: str
    contract_version: int
    min_contract_version: int
    app_version: str
    last_seen: datetime
