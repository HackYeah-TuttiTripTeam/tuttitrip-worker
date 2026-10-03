"""Queue registration: contract queues only, stale queues dropped safely."""

from dbos import DBOS, DBOSClient, EnqueueOptions, WorkflowSerializationFormat

from tuttitrip_worker.contracts import CONTRACT_VERSION, Queue, Workflow
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.dbos.runtime import register_queues


def queue_names() -> set[str]:
    return {queue.name for queue in DBOS.list_queues()}


def test_stale_queue_is_deleted(dbos: Settings) -> None:
    del dbos
    DBOS.register_queue("system", worker_concurrency=8)
    register_queues()
    assert queue_names() == {queue.value for queue in Queue}


def test_stale_queue_with_queued_work_is_kept(
    client: DBOSClient, dbos: Settings
) -> None:
    del dbos
    DBOS.register_queue("planning", worker_concurrency=8)
    # Another app version: this worker never dequeues it, so it stays ENQUEUED.
    options: EnqueueOptions = {
        "workflow_name": Workflow.PING.value,
        "queue_name": "planning",
        "app_version": "other",
        "serialization_type": WorkflowSerializationFormat.PORTABLE,
    }
    client.enqueue(options, {"contract_version": CONTRACT_VERSION, "message": "x"})
    register_queues()
    assert "planning" in queue_names()
