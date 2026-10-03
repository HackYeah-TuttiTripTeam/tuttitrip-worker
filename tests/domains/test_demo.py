"""Demo domain: the daily reset call to the backend's internal endpoint."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
from dbos import DBOS

from tuttitrip_worker.contracts import SCHEDULE_TIMEZONE, SCHEDULED_WORKFLOWS
from tuttitrip_worker.demo import steps, workflows
from tuttitrip_worker.demo.logic.target import RESET_PATH, api_base_url
from tuttitrip_worker.demo.schemas import DemoResetResult
from tuttitrip_worker.demo.services.backend_client import (
    DemoResetError,
    request_reset,
)
from tuttitrip_worker.main import SCHEDULES, apply_schedules
from tuttitrip_worker.shared.config.settings import Settings, get_settings

URL = f"http://tuttitrip-api-develop:8000{RESET_PATH}"
SECRET = "s3cret-value"  # ruff: ignore[hardcoded-password-string]  # a test double


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler)


@pytest.mark.parametrize(
    ("environment", "override", "expected"),
    [
        ("main", "", "http://tuttitrip-api:8000"),
        ("develop", "", "http://tuttitrip-api-develop:8000"),
        ("feature-x", "", "http://tuttitrip-api-feature-x:8000"),
        ("local", "", "http://localhost:8000"),
        ("develop", "http://api.test:9000/", "http://api.test:9000"),
    ],
)
def test_api_base_url_follows_the_container_naming(
    environment: str, override: str, expected: str
) -> None:
    assert api_base_url(environment, override) == expected


def test_request_reset_sends_the_secret_as_bearer_token() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"status": "reset", "trips": 4})

    async def run() -> DemoResetResult:
        async with _client(httpx.MockTransport(handler)) as client:
            return await request_reset(client, URL, SECRET)

    assert asyncio.run(run()) == DemoResetResult(status="reset", trips=4)
    assert seen[0].method == "POST"
    assert str(seen[0].url) == URL
    assert seen[0].headers["Authorization"] == f"Bearer {SECRET}"


def test_request_reset_reports_a_disabled_demo() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "disabled"})

    async def run() -> DemoResetResult:
        async with _client(httpx.MockTransport(handler)) as client:
            return await request_reset(client, URL, SECRET)

    assert asyncio.run(run()) == DemoResetResult(status="disabled", trips=0)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404, text=SECRET),
        httpx.Response(401),
        httpx.Response(500),
        httpx.Response(200, content=b"not json"),
        httpx.Response(200, content=json.dumps({"status": "?"}).encode()),
    ],
)
def test_request_reset_fails_without_leaking_the_secret(
    response: httpx.Response,
) -> None:
    async def run() -> None:
        transport = httpx.MockTransport(lambda _request: response)
        async with _client(transport) as client:
            await request_reset(client, URL, SECRET)

    with pytest.raises(DemoResetError) as raised:
        asyncio.run(run())
    assert SECRET not in str(raised.value)


def test_request_reset_wraps_transport_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        msg = f"boom {SECRET}"
        raise httpx.ConnectError(msg, request=request)

    async def run() -> None:
        async with _client(httpx.MockTransport(handler)) as client:
            await request_reset(client, URL, SECRET)

    with pytest.raises(DemoResetError, match="ConnectError") as raised:
        asyncio.run(run())
    assert SECRET not in str(raised.value)


def _run_workflow() -> None:
    asyncio.run(workflows.reset_demo_account(datetime.now(UTC), None))


def test_workflow_calls_the_step_when_configured(
    dbos: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    del dbos
    monkeypatch.setenv("TUTTITRIP_DEMO__RESET_SECRET", SECRET)
    get_settings.cache_clear()
    calls: list[str] = []

    async def fake_step() -> dict[str, Any]:
        calls.append("called")
        return {"status": "reset", "trips": 4}

    monkeypatch.setattr(steps, "reset_demo_account", fake_step)
    with caplog.at_level(logging.INFO, logger="tuttitrip-worker"):
        _run_workflow()
    assert calls == ["called"]
    assert "reset (4 trips)" in caplog.text
    assert SECRET not in caplog.text


def test_workflow_does_nothing_without_a_secret(
    dbos: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    del dbos

    async def fake_step() -> dict[str, Any]:
        pytest.fail("the backend must not be called")

    monkeypatch.setattr(steps, "reset_demo_account", fake_step)
    _run_workflow()


def test_workflow_treats_a_disabled_demo_as_success(
    dbos: Settings, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    del dbos
    monkeypatch.setenv("TUTTITRIP_DEMO__RESET_SECRET", SECRET)
    get_settings.cache_clear()

    async def fake_step() -> dict[str, Any]:
        return {"status": "disabled"}

    monkeypatch.setattr(steps, "reset_demo_account", fake_step)
    with caplog.at_level(logging.INFO, logger="tuttitrip-worker"):
        _run_workflow()
    assert "disabled" in caplog.text


def test_schedule_runs_daily_at_four_warsaw_time(dbos: Settings) -> None:
    del dbos
    assert SCHEDULE_TIMEZONE == "Europe/Warsaw"
    assert SCHEDULED_WORKFLOWS["reset_demo_account"] == "0 0 4 * * *"
    assert set(SCHEDULES) == set(SCHEDULED_WORKFLOWS)
    apply_schedules()
    names = {schedule["schedule_name"] for schedule in DBOS.list_schedules()}
    assert "reset_demo_account" in names
