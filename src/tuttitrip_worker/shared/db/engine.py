"""Async SQLAlchemy engine for the application database (psycopg 3).

Only DBOS steps use it: database access is I/O and must never run directly
in a workflow body. The worker never runs DDL; see ``tables.py``.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from tuttitrip_worker.shared.config.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Create the process-wide engine on first use.

    Returns:
        The cached async engine.
    """
    settings = get_settings()
    url = make_url(settings.worker_database_url.get_secret_value())
    return create_async_engine(
        # The env file names plain `postgresql://`; use async psycopg 3.
        url.set(drivername="postgresql+psycopg"),
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
    )


@asynccontextmanager
async def transaction() -> AsyncGenerator[AsyncConnection]:
    """Open a connection with a transaction that commits on success.

    Yields:
        A connection inside ``BEGIN ... COMMIT`` (rolled back on error).
    """
    async with get_engine().begin() as connection:
        yield connection


async def dispose_engine() -> None:
    """Close pooled connections (graceful shutdown)."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
