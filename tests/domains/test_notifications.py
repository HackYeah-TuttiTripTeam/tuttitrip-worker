"""Notifications: the worker's inserts, the plan_ready producer and the daily purge."""

import asyncio
from collections.abc import AsyncGenerator, Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any, get_args
from uuid import UUID, uuid4

import pytest
from dbos import DBOSClient
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel
from sqlalchemy import ClauseElement, Connection, Engine, create_engine, insert, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.base import Executable

from tests.helpers import enqueue
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    SCHEDULED_WORKFLOWS,
    NotificationActionCode,
    NotificationDraft,
    NotificationType,
    Workflow,
)
from tuttitrip_worker.main import SCHEDULES
from tuttitrip_worker.notifications import steps as purge_steps
from tuttitrip_worker.notifications.logic.retention import BATCH_SIZE, cutoffs
from tuttitrip_worker.notifications.workflows import purge_notifications
from tuttitrip_worker.planning import workflows as planning_workflows
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.db import job_results, notifications
from tuttitrip_worker.shared.db.tables import notifications as table
from tuttitrip_worker.shared.llm.models import catalog

# The backend's `NotificationType` and `NotificationActionCode`
# (tuttitrip.notifications.schemas, backend#165/#188). A type or code the worker
# adds without the backend knowing it makes this test fail.
BACKEND_TYPES = {
    "member_joined",
    "veto_added",
    "proposal_waiting",
    "budget_approval_waiting",
    "plan_ready",
}
BACKEND_ACTIONS = {
    "open_trip",
    "open_people",
    "open_plan",
    "approve_proposal",
    "reject_proposal",
    "approve_budget",
    "reject_budget",
}
NOW = datetime(2026, 10, 4, 3, 30, tzinfo=UTC)


def test_the_worker_writes_only_types_and_actions_the_backend_knows() -> None:
    assert set(get_args(NotificationType)) <= BACKEND_TYPES
    assert set(get_args(NotificationActionCode)) <= BACKEND_ACTIONS


def draft(key: str = "plan_ready:t:w", **changes: object) -> NotificationDraft:
    base: dict[str, object] = {
        "type": "plan_ready",
        "trip_id": uuid4(),
        "params": {"destination": "Kraków"},
        "actions": ["open_plan"],
        "dedupe_key": key,
    }
    return NotificationDraft.model_validate({**base, **changes})


def test_an_unknown_type_or_a_bad_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        draft(type="veto_added")
    with pytest.raises(ValidationError):
        draft(actions=["open_the_vault"])
    with pytest.raises(ValidationError):
        draft(key="")
    with pytest.raises(ValidationError):
        draft(key="k" * 256)


def sql(statement: ClauseElement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_the_insert_does_nothing_on_a_repeated_key() -> None:
    text = sql(notifications.build_notification_insert("auth0|u1", draft()))
    assert "INSERT INTO notifications" in text
    assert "ON CONFLICT (user_sub, dedupe_key) DO NOTHING" in text
    assert "UPDATE" not in text  # the worker has no UPDATE grant


def test_the_id_follows_the_user_and_the_key() -> None:
    first = notifications.notification_id("auth0|u1", "plan_ready:t:w")
    assert first == notifications.notification_id("auth0|u1", "plan_ready:t:w")
    assert first != notifications.notification_id("auth0|u2", "plan_ready:t:w")
    assert first != notifications.notification_id("auth0|u1", "plan_ready:t:w2")


def test_the_insert_carries_buttons_as_codes_without_urls() -> None:
    statement = notifications.build_notification_insert("auth0|u1", draft())
    params = statement.compile(dialect=postgresql.dialect()).params
    assert params["actions"] == [{"code": "open_plan", "params": {}}]
    assert params["params"] == {"destination": "Kraków"}
    assert params["type"] == "plan_ready"


# --- plan_ready from generate_trip_plan ---------------------------------------------

PLAN = {"destination": "Kraków", "days": 2, "highlights": ["Wawel"]}


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def fake_notify(user_sub: str, body: dict[str, Any]) -> None:
        calls.append((user_sub, body))

    async def fake_save(workflow_id: str, name: str, result: dict[str, Any]) -> None:
        del workflow_id, name, result

    monkeypatch.setattr(planning_workflows, "notify_user", fake_notify)
    monkeypatch.setattr(job_results, "save_job_result", fake_save)
    return calls


def plan_payload() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "trip_id": str(uuid4()),
        "request": "Weekend w Krakowie",
    }


def test_the_user_who_started_the_plan_is_told_when_it_is_ready(
    client: DBOSClient, dbos: Settings, sent: list[tuple[str, dict[str, Any]]]
) -> None:
    body = plan_payload()
    with catalog.override(TestModel(custom_output_args=PLAN)):
        handle = enqueue(
            client, dbos, Workflow.GENERATE_TRIP_PLAN, body, user="auth0|host"
        )
        handle.get_result()

    assert len(sent) == 1
    user, note = sent[0]
    assert user == "auth0|host"
    assert note["type"] == "plan_ready"
    assert note["trip_id"] == body["trip_id"]
    assert note["actions"] == ["open_plan"]
    assert note["params"] == {"destination": "Kraków"}
    assert (
        note["dedupe_key"] == f"plan_ready:{body['trip_id']}:{handle.get_workflow_id()}"
    )


def test_nobody_is_told_when_the_job_has_no_user(
    client: DBOSClient, dbos: Settings, sent: list[tuple[str, dict[str, Any]]]
) -> None:
    with catalog.override(TestModel(custom_output_args=PLAN)):
        enqueue(client, dbos, Workflow.GENERATE_TRIP_PLAN, plan_payload()).get_result()
    assert sent == []


# --- the daily purge -------------------------------------------------------------


@pytest.fixture
def engine() -> Iterator[Engine]:
    sqlite = create_engine("sqlite://")
    table.metadata.create_all(sqlite, tables=[table])
    yield sqlite
    sqlite.dispose()


class SqliteConnection:
    """Runs the real statements on SQLite behind the async step interface."""

    def __init__(self, connection: Connection) -> None:
        self.connection = connection

    async def execute(self, statement: Executable) -> object:
        return self.connection.execute(statement)


@pytest.fixture
def purge_db(engine: Engine, monkeypatch: pytest.MonkeyPatch) -> Engine:
    @asynccontextmanager
    async def transaction() -> AsyncGenerator[SqliteConnection]:
        with engine.begin() as connection:
            yield SqliteConnection(connection)

    monkeypatch.setattr(purge_steps, "transaction", transaction)
    return engine


def add(engine: Engine, count: int, *, age_days: int, read: bool) -> list[UUID]:
    ids = [uuid4() for _ in range(count)]
    created = NOW - timedelta(days=age_days)
    rows = [
        {
            "id": row_id,
            "user_sub": "auth0|u1",
            "type": "plan_ready",
            "trip_id": None,
            "params": {},
            "actions": [],
            "dedupe_key": str(row_id),
            "read_at": created if read else None,
            "created_at": created,
        }
        for row_id in ids
    ]
    with engine.begin() as connection:
        connection.execute(insert(table), rows)
    return ids


def remaining(engine: Engine) -> set[UUID]:
    with engine.connect() as connection:
        return set(connection.execute(select(table.c.id)).scalars())


def run_purge(dbos: Settings) -> dict[str, int]:
    del dbos
    return asyncio.run(purge_notifications(NOW, None))


def test_cutoffs_are_90_days_for_read_and_180_for_all() -> None:
    limits = cutoffs(NOW)
    assert NOW - limits.read == timedelta(days=90)
    assert NOW - limits.any == timedelta(days=180)


def test_a_read_notification_after_91_days_goes_and_an_unread_one_at_100_stays(
    dbos: Settings, purge_db: Engine
) -> None:
    old_read = add(purge_db, 1, age_days=91, read=True)
    old_unread = add(purge_db, 1, age_days=100, read=False)
    fresh_read = add(purge_db, 1, age_days=30, read=True)
    ancient_unread = add(purge_db, 1, age_days=181, read=False)

    result = run_purge(dbos)

    assert result == {"deleted": 2, "batches": 1}
    assert remaining(purge_db) == {*old_unread, *fresh_read}
    assert not remaining(purge_db) & {*old_read, *ancient_unread}


def test_5000_old_rows_go_in_batches_of_1000(dbos: Settings, purge_db: Engine) -> None:
    add(purge_db, 5000, age_days=200, read=False)
    keep = add(purge_db, 3, age_days=1, read=False)

    result = run_purge(dbos)

    assert BATCH_SIZE == 1000
    assert result == {
        "deleted": 5000,
        "batches": 6,
    }  # five full batches and an empty one
    assert remaining(purge_db) == set(keep)


def test_the_batch_statement_is_one_limited_delete() -> None:
    text = sql(purge_steps.build_purge_batch(NOW, NOW))
    assert text.startswith("DELETE FROM notifications WHERE notifications.id IN")
    assert "LIMIT" in text


def test_the_purge_is_scheduled_once_a_day() -> None:
    assert "purge_notifications" in SCHEDULES
    assert SCHEDULED_WORKFLOWS["purge_notifications"] == "0 30 3 * * *"
