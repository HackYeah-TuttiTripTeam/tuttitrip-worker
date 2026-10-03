"""Read-only access to DBOS system databases through the official DBOSClient."""

from dataclasses import dataclass
from typing import Protocol

from dbos import DBOSClient, WorkflowStatus

STATUSES = (
    "ENQUEUED",
    "PENDING",
    "SUCCESS",
    "ERROR",
    "CANCELLED",
    "MAX_RECOVERY_ATTEMPTS_EXCEEDED",
)


@dataclass(frozen=True)
class WorkflowRow:
    """One workflow as listed by the dashboard."""

    workflow_id: str
    name: str
    status: str
    queue_name: str | None
    app_version: str | None
    created_at_ms: int | None
    updated_at_ms: int | None
    recovery_attempts: int | None
    executor_id: str | None
    error: str | None = None


@dataclass(frozen=True)
class QueueRow:
    """A registered queue with its current backlog."""

    name: str
    concurrency: int | None
    worker_concurrency: int | None
    enqueued: int
    pending: int


@dataclass(frozen=True)
class StepRow:
    """One recorded step of a workflow."""

    function_id: int
    function_name: str
    error: str | None
    child_workflow_id: str | None
    started_at_ms: int | None
    completed_at_ms: int | None


class DashboardSource(Protocol):
    """What the HTTP layer needs; the real one reads DBOS, tests fake it."""

    def envs(self) -> list[str]:
        """Return the environment names in display order."""
        ...

    def workflows(
        self, env: str, *, status: str | None, name: str | None, limit: int
    ) -> list[WorkflowRow]:
        """Return the newest workflows of an environment."""
        ...

    def queues(self, env: str) -> list[QueueRow]:
        """Return the queues of an environment with their backlog."""
        ...

    def workflow(self, env: str, workflow_id: str) -> WorkflowRow | None:
        """Return one workflow with its error, if it exists."""
        ...

    def steps(self, env: str, workflow_id: str) -> list[StepRow]:
        """Return the recorded steps of a workflow."""
        ...


def _error_text(error: object) -> str | None:
    if error is None:
        return None
    text = str(error) or type(error).__name__
    return (
        f"{type(error).__name__}: {text}" if isinstance(error, BaseException) else text
    )


def _row(status: WorkflowStatus, *, with_error: bool = False) -> WorkflowRow:
    return WorkflowRow(
        workflow_id=status.workflow_id,
        name=status.name,
        status=status.status,
        queue_name=status.queue_name,
        app_version=status.app_version,
        created_at_ms=status.created_at,
        updated_at_ms=status.updated_at,
        recovery_attempts=status.recovery_attempts,
        executor_id=status.executor_id,
        error=_error_text(status.error) if with_error else None,
    )


class DbosSource:
    """One lazily connected DBOSClient per environment (read-only use only)."""

    def __init__(self, databases: dict[str, str]) -> None:
        self._databases = databases
        self._clients: dict[str, DBOSClient] = {}

    def _client(self, env: str) -> DBOSClient:
        if env not in self._databases:
            raise KeyError(env)
        if env not in self._clients:
            self._clients[env] = DBOSClient(
                system_database_url=self._databases[env], system_database_pool_size=2
            )
        return self._clients[env]

    def close(self) -> None:
        """Close every database connection pool."""
        for client in self._clients.values():
            client.destroy()
        self._clients.clear()

    def envs(self) -> list[str]:
        """Return the configured environments.

        Returns:
            Environment names in configuration order.
        """
        return list(self._databases)

    def workflows(
        self, env: str, *, status: str | None, name: str | None, limit: int
    ) -> list[WorkflowRow]:
        """List the newest workflows without loading inputs or outputs.

        Args:
            env: Environment name.
            status: Only this status, if given.
            name: Only this workflow name, if given.
            limit: Maximum number of rows.

        Returns:
            Workflows, newest first.
        """
        found = self._client(env).list_workflows(
            status=status,
            name=name,
            limit=limit,
            sort_desc=True,
            load_input=False,
            load_output=False,
        )
        return [_row(s) for s in found]

    def queues(self, env: str) -> list[QueueRow]:
        """List queues with the number of enqueued and running workflows.

        Args:
            env: Environment name.

        Returns:
            Queues sorted by name.
        """
        client = self._client(env)
        waiting = client.list_queued_workflows(load_input=False, load_output=False)
        counts: dict[str, dict[str, int]] = {}
        for s in waiting:
            per_queue = counts.setdefault(s.queue_name or "-", {})
            per_queue[s.status] = per_queue.get(s.status, 0) + 1
        names = {q.name: q for q in client.list_queues()}
        rows = []
        for queue_name in sorted(set(names) | set(counts)):
            queue = names.get(queue_name)
            backlog = counts.get(queue_name, {})
            rows.append(
                QueueRow(
                    name=queue_name,
                    concurrency=getattr(queue, "concurrency", None),
                    worker_concurrency=getattr(queue, "worker_concurrency", None),
                    enqueued=backlog.get("ENQUEUED", 0),
                    pending=backlog.get("PENDING", 0),
                )
            )
        return rows

    def workflow(self, env: str, workflow_id: str) -> WorkflowRow | None:
        """Read one workflow including its error (no inputs or outputs).

        Args:
            env: Environment name.
            workflow_id: Workflow id.

        Returns:
            The workflow, or None if it does not exist.
        """
        found = self._client(env).list_workflows(
            workflow_ids=[workflow_id], load_input=False, load_output=True
        )
        return _row(found[0], with_error=True) if found else None

    def steps(self, env: str, workflow_id: str) -> list[StepRow]:
        """List the steps of a workflow (outputs are not loaded).

        Args:
            env: Environment name.
            workflow_id: Workflow id.

        Returns:
            Steps in execution order.
        """
        found = self._client(env).list_workflow_steps(workflow_id, load_output=False)
        return [
            StepRow(
                function_id=s["function_id"],
                function_name=s["function_name"],
                error=_error_text(s["error"]),
                child_workflow_id=s["child_workflow_id"],
                started_at_ms=s["started_at_epoch_ms"],
                completed_at_ms=s["completed_at_epoch_ms"],
            )
            for s in found
        ]
