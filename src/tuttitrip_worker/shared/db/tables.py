"""Read/write mappings of the tables the worker writes (no DDL, ever).

The backend owns the schema: its Alembic migrations create these tables
(``migrations/versions/*_worker_shared_tables_and_embeddings.py`` in
``tuttitrip-backend``) and ``deploy/worker-grants.sql`` there gives the role
``tuttitrip_worker`` SELECT on what it reads and SELECT/INSERT/UPDATE/DELETE
on these three tables only. The worker never calls ``create_all``, never runs
migrations and never issues DDL (``tests/test_no_ddl.py``).

The definitions only describe columns for SQLAlchemy Core statements. Keep
them in sync with the backend migrations; server defaults (``created_at``,
``last_seen``) are omitted on purpose.
"""

from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB

# Never passed to create_all/drop_all; it only groups the mappings.
metadata = MetaData()

embeddings = Table(
    "embeddings",
    metadata,
    # No unique key in the schema: the worker derives a deterministic id from
    # (source, model, content), so re-running a job upserts the same rows.
    Column("id", Uuid, primary_key=True),
    Column("source_kind", String(50), nullable=False),
    Column("source_id", String(200), nullable=False),
    Column("content", Text, nullable=False),
    Column("model", String(200), nullable=False),
    Column("embedding", Vector(), nullable=False),
)

job_results = Table(
    "job_results",
    metadata,
    Column("workflow_id", String(300), primary_key=True),
    Column("workflow_name", String(100), nullable=False),
    Column("contract_version", Integer, nullable=False),
    Column("result", JSONB, nullable=False),
)

worker_heartbeats = Table(
    "worker_heartbeats",
    metadata,
    Column("worker_id", String(200), primary_key=True),
    Column("env", String(63), nullable=False),
    Column("contract_version", Integer, nullable=False),
    Column("min_contract_version", Integer, nullable=False),
    Column("app_version", String(200), nullable=False),
    Column("last_seen", DateTime(timezone=True), nullable=False),
)
