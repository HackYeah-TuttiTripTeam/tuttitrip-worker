"""Parsing of TUTTITRIP_DBOS_DASHBOARD_DATABASES."""

import pytest

from tuttitrip_dbos_dashboard.config import (
    DATABASES_ENV,
    ConfigError,
    load_config,
    parse_databases,
)


def test_pairs_keep_their_order() -> None:
    parsed = parse_databases(
        "main=postgresql://u:p@h:5432/a, develop=postgresql://u:p@h:5432/b"
    )
    assert list(parsed) == ["main", "develop"]
    assert parsed["develop"].endswith("/b")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "main",
        "Main=postgresql://h/a",
        "main=mysql://h/a",
        "main=postgresql://h/a,main=postgresql://h/b",
    ],
    ids=["empty", "no-url", "uppercase-env", "not-postgres", "duplicate"],
)
def test_bad_values_are_rejected(raw: str) -> None:
    with pytest.raises(ConfigError):
        parse_databases(raw)


def test_errors_never_echo_the_url() -> None:
    with pytest.raises(ConfigError) as error:
        parse_databases("main=mysql://user:secret@h/a")
    assert "secret" not in str(error.value)


def test_load_config_reads_port_and_databases() -> None:
    config = load_config(
        {
            DATABASES_ENV: "main=sqlite:///x.db",
            "TUTTITRIP_DBOS_DASHBOARD_PORT": "9000",
        }
    )
    assert config.port == 9000
    assert config.databases == {"main": "sqlite:///x.db"}
