"""Dashboard settings from the environment (written by the backend's admin setup)."""

import os
import re
from dataclasses import dataclass

DATABASES_ENV = "TUTTITRIP_DBOS_DASHBOARD_DATABASES"
PORT_ENV = "TUTTITRIP_DBOS_DASHBOARD_PORT"
DEFAULT_PORT = 8080
_ENV_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,48}$")


class ConfigError(ValueError):
    """The dashboard environment variables are missing or malformed."""


@dataclass(frozen=True)
class DashboardConfig:
    """Which DBOS system databases to show and where to listen."""

    databases: dict[str, str]
    port: int = DEFAULT_PORT


def parse_databases(raw: str) -> dict[str, str]:
    """Parse ``env=postgresql://...`` pairs separated by commas, keeping order.

    Args:
        raw: Value of ``TUTTITRIP_DBOS_DASHBOARD_DATABASES``.

    Returns:
        Environment name -> system database URL.
    """
    databases: dict[str, str] = {}
    for item in filter(None, (part.strip() for part in raw.split(","))):
        name, sep, url = item.partition("=")
        name = name.strip()
        if not sep or not _ENV_NAME.match(name):
            msg = f"bad entry in {DATABASES_ENV}: expected <env>=<url>"
            raise ConfigError(msg)
        if not url.startswith(("postgresql://", "postgresql+psycopg://", "sqlite:///")):
            msg = f"bad URL for env {name!r} in {DATABASES_ENV}"
            raise ConfigError(msg)
        if name in databases:
            msg = f"env {name!r} listed twice in {DATABASES_ENV}"
            raise ConfigError(msg)
        databases[name] = url
    if not databases:
        msg = f"{DATABASES_ENV} is empty"
        raise ConfigError(msg)
    return databases


def load_config(environ: dict[str, str] | None = None) -> DashboardConfig:
    """Build the configuration from environment variables.

    Args:
        environ: Variables to read; defaults to ``os.environ``.

    Returns:
        The dashboard configuration.
    """
    env = dict(os.environ) if environ is None else environ
    port = int(env.get(PORT_ENV, str(DEFAULT_PORT)))
    return DashboardConfig(
        databases=parse_databases(env.get(DATABASES_ENV, "")), port=port
    )
