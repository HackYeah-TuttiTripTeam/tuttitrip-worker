"""System domain: ping through the queue, heartbeat rows."""

from dbos import DBOSClient

from tests.helpers import enqueue
from tuttitrip_worker.contracts import (
    APPLICATION_NAME,
    CONTRACT_VERSION,
    PingOutput,
    Workflow,
)
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.system.steps import current_heartbeat


def test_ping_echoes_through_the_default_queue(
    client: DBOSClient, dbos: Settings
) -> None:
    payload = {"contract_version": CONTRACT_VERSION, "message": "hello"}
    handle = enqueue(client, dbos, Workflow.PING, payload, workflow_id="ping-1")
    output = PingOutput.model_validate(handle.get_result())
    assert output.message == "hello"
    assert output.worker_app_version == dbos.application_version == "local"


def test_same_workflow_id_runs_once(client: DBOSClient, dbos: Settings) -> None:
    payload = {"contract_version": CONTRACT_VERSION, "message": "first"}
    first = enqueue(client, dbos, Workflow.PING, payload, workflow_id="dup")
    assert first.get_result()["message"] == "first"
    again = {"contract_version": CONTRACT_VERSION, "message": "second"}
    second = enqueue(client, dbos, Workflow.PING, again, workflow_id="dup")
    # The backend's deterministic workflow ids make retries idempotent.
    assert second.get_result()["message"] == "first"


def test_heartbeat_identifies_the_environment_worker() -> None:
    beat = current_heartbeat()
    assert beat.worker_id == f"{APPLICATION_NAME}-local"
    assert beat.env == "local"
    assert beat.min_contract_version <= CONTRACT_VERSION <= beat.contract_version
    assert beat.last_seen.tzinfo is not None
