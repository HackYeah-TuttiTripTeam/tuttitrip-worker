"""DbosSource against a real DBOS runtime on SQLite (as the backend enqueues)."""

from dbos import DBOSClient

from tests.helpers import enqueue
from tuttitrip_dbos_dashboard.source import DbosSource
from tuttitrip_worker.contracts import CONTRACT_VERSION, Workflow
from tuttitrip_worker.shared.config.settings import Settings


def test_lists_a_finished_ping(
    client: DBOSClient, dbos: Settings, sqlite_url: str
) -> None:
    payload = {"contract_version": CONTRACT_VERSION, "message": "hi"}
    enqueue(client, dbos, Workflow.PING, payload, workflow_id="dash-1").get_result()
    source = DbosSource({"local": sqlite_url})
    try:
        check(source)
    finally:
        source.close()


def check(source: DbosSource) -> None:
    rows = source.workflows("local", status="SUCCESS", name="ping", limit=10)
    assert [r.workflow_id for r in rows] == ["dash-1"]
    assert rows[0].queue_name == "default"
    assert rows[0].app_version == "local"

    detail = source.workflow("local", "dash-1")
    assert detail is not None
    assert detail.status == "SUCCESS"
    assert detail.error is None
    assert source.workflow("local", "missing") is None
    assert isinstance(source.steps("local", "dash-1"), list)

    queues = {q.name: q for q in source.queues("local")}
    assert {"default", "local_llm", "openrouter"} <= set(queues)
    assert queues["default"].enqueued == 0
