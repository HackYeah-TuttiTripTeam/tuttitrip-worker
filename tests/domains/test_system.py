"""System domain: ping through the queue, heartbeat rows."""

import pytest
from dbos import DBOSClient, EnqueueOptions, WorkflowSerializationFormat

from tests.helpers import enqueue
from tuttitrip_worker.contracts import (
    APPLICATION_NAME,
    CONTRACT_VERSION,
    PingOutput,
    Workflow,
)
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.dbos.runtime import LEGACY_QUEUES
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


@pytest.mark.parametrize("queue", sorted(LEGACY_QUEUES))
def test_legacy_queue_names_are_still_served(
    client: DBOSClient, dbos: Settings, queue: str
) -> None:
    # The backend mirror of contract v1 still enqueues on the old names.
    options: EnqueueOptions = {
        "workflow_name": Workflow.PING.value,
        "queue_name": queue,
        "app_version": dbos.application_version,
        "serialization_type": WorkflowSerializationFormat.PORTABLE,
    }
    payload = {"contract_version": CONTRACT_VERSION, "message": queue}
    assert client.enqueue(options, payload).get_result()["message"] == queue
