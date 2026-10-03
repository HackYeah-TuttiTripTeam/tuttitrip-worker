"""Test helpers that enqueue exactly like the backend does."""

from typing import Any

from dbos import DBOSClient, EnqueueOptions, WorkflowHandle, WorkflowSerializationFormat

from tuttitrip_worker.contracts import WORKFLOWS, Workflow
from tuttitrip_worker.shared.config.settings import Settings


def enqueue(
    client: DBOSClient,
    settings: Settings,
    name: Workflow,
    payload: dict[str, Any],
    *,
    workflow_id: str | None = None,
) -> WorkflowHandle[Any]:
    """Enqueue a contract workflow the way the backend's DBOSClient does."""
    options: EnqueueOptions = {
        "workflow_name": name.value,
        "queue_name": WORKFLOWS[name].queue.value,
        "app_version": settings.application_version,
        "serialization_type": WorkflowSerializationFormat.PORTABLE,
    }
    if workflow_id is not None:
        options["workflow_id"] = workflow_id
    return client.enqueue(options, payload)
