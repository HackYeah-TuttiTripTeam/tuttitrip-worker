"""Shared pytest configuration.

* No test may call a real model (``ALLOW_MODEL_REQUESTS = False``).
* Settings never come from the developer's ``.env`` or ``TUTTITRIP_*`` vars.
* ``dbos`` runs a real DBOS runtime on a throwaway SQLite system database,
  so workflow tests need no Postgres (steps that touch Postgres are stubbed).
"""

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from dbos import DBOS, DBOSClient
from pydantic_ai import models

from tuttitrip_worker import main as worker_main
from tuttitrip_worker.contracts import APPLICATION_NAME
from tuttitrip_worker.shared.config.settings import Settings, get_settings
from tuttitrip_worker.shared.dbos.runtime import build_config, register_queues

# Safety net: fail loudly if any test tries to call a real LLM provider.
models.ALLOW_MODEL_REQUESTS = False

# Importing main registers every workflow before any DBOS instance launches.
assert worker_main.WORKFLOWS


@pytest.fixture(autouse=True)
def isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Generator[None]:
    for key in list(os.environ):
        if key.startswith(("TUTTITRIP_", "DBOS")) or key == "OPENROUTER_API_KEY":
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)  # no .env here
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sqlite_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'dbos.sqlite'}"


@pytest.fixture
def dbos(sqlite_url: str) -> Generator[Settings]:
    """A launched DBOS runtime with every queue registered."""
    settings = get_settings()
    config = build_config(settings)
    config["system_database_url"] = sqlite_url
    config["run_migrations"] = True  # a fresh SQLite file; never on Postgres
    DBOS.destroy()
    DBOS(config=config)
    DBOS.launch()
    register_queues()
    yield settings
    DBOS.destroy()


@pytest.fixture
def client(dbos: Settings, sqlite_url: str) -> Generator[DBOSClient]:
    """A client configured like the backend's (application name, version)."""
    del dbos
    dbos_client = DBOSClient(
        system_database_url=sqlite_url, application_name=APPLICATION_NAME
    )
    yield dbos_client
    dbos_client.destroy()
