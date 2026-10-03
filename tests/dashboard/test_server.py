"""HTTP layer of the dashboard with a fake data source (no database)."""

import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from http.server import ThreadingHTTPServer

import pytest

from tuttitrip_dbos_dashboard.server import make_handler
from tuttitrip_dbos_dashboard.source import QueueRow, StepRow, WorkflowRow

EVIL = '<script>alert("x")</script>'


def _wf(workflow_id: str, status: str, error: str | None = None) -> WorkflowRow:
    return WorkflowRow(
        workflow_id=workflow_id,
        name=f"ping{EVIL}",
        status=status,
        queue_name="default",
        app_version="main",
        created_at_ms=1_700_000_000_000,
        updated_at_ms=1_700_000_001_500,
        recovery_attempts=1,
        executor_id="local",
        error=error,
    )


class FakeSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.env_names = ["main", "develop"]
        self.backlog = [QueueRow("default", None, 8, enqueued=3, pending=1)]
        self.errored = _wf("wf-2", "ERROR", error=EVIL)
        self.step_rows = [
            StepRow(1, "run_model", EVIL, None, 1_700_000_000_000, 1_700_000_000_250)
        ]

    def envs(self) -> list[str]:
        return self.env_names

    def workflows(
        self, env: str, *, status: str | None, name: str | None, limit: int
    ) -> list[WorkflowRow]:
        self.calls.append(("workflows", (env, status, name, limit)))
        return [_wf("wf-1", "SUCCESS"), _wf("wf-2", "ERROR")]

    def queues(self, env: str) -> list[QueueRow]:
        del env
        return self.backlog

    def workflow(self, env: str, workflow_id: str) -> WorkflowRow | None:
        del env
        return self.errored if workflow_id == self.errored.workflow_id else None

    def steps(self, env: str, workflow_id: str) -> list[StepRow]:
        del env, workflow_id
        return self.step_rows


@pytest.fixture
def fake() -> FakeSource:
    return FakeSource()


@pytest.fixture
def base_url(fake: FakeSource) -> Generator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(fake))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url: str, headers: dict[str, str] | None = None) -> tuple[int, str]:
    request = urllib.request.Request(url, headers=headers or {})  # ruff: ignore[suspicious-url-open-usage]
    try:
        with urllib.request.urlopen(request) as response:  # ruff: ignore[suspicious-url-open-usage]
            return int(response.status), bytes(response.read()).decode()
    except urllib.error.HTTPError as error:
        with error:
            return int(error.code), bytes(error.read()).decode()


def test_healthz(base_url: str) -> None:
    assert get(f"{base_url}/healthz") == (200, "ok")


def test_overview_lists_queues_and_workflows_escaped(base_url: str) -> None:
    status, body = get(base_url + "/", {"X-Auth-Request-Email": "admin@example.com"})
    assert status == 200
    assert "wf-1" in body
    assert "wf-2" in body
    assert "<td>default</td><td>3</td><td>1</td>" in body
    assert "Zalogowano: admin@example.com" in body
    assert EVIL not in body
    assert "&lt;script&gt;" in body


def test_overview_passes_sanitised_filters(base_url: str, fake: FakeSource) -> None:
    get(f"{base_url}/?env=develop&status=ERROR&name=ping&limit=9999")
    get(f"{base_url}/?env=develop&status=DROP&limit=abc")
    assert fake.calls == [
        ("workflows", ("develop", "ERROR", "ping", 500)),
        ("workflows", ("develop", None, None, 100)),
    ]


def test_unknown_env_is_404(base_url: str) -> None:
    assert get(f"{base_url}/?env=prod")[0] == 404


def test_workflow_page_shows_error_and_steps_escaped(base_url: str) -> None:
    status, body = get(f"{base_url}/workflow?env=main&id=wf-2")
    assert status == 200
    assert "run_model" in body
    assert "0.2 s" in body
    assert EVIL not in body


def test_missing_workflow_is_404(base_url: str) -> None:
    assert get(f"{base_url}/workflow?env=main&id=nope")[0] == 404
    assert get(f"{base_url}/workflow?env=main")[0] == 404


def test_unknown_path_is_404(base_url: str) -> None:
    assert get(f"{base_url}/admin")[0] == 404


def test_pages_forbid_scripts(base_url: str) -> None:
    with urllib.request.urlopen(base_url + "/") as response:  # ruff: ignore[suspicious-url-open-usage]
        assert "default-src 'none'" in response.headers["Content-Security-Policy"]
