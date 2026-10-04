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
from sqlalchemy import (
    ARRAY,
    JSON,
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Double,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB

# Never passed to create_all/drop_all; it only groups the mappings.
metadata = MetaData()

# Read-only: SELECT on `plan_versions` comes from the backend's
# `deploy/worker-grants.sql` (tuttitrip-backend#73). Only the columns used.
plan_versions = Table(
    "plan_versions",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("trip_id", Uuid, nullable=False),
    Column("result", JSONB, nullable=False),
)

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

# Read-only for the worker (SELECT grant): texts pasted for a trip, kind
# `plan` or `offer`.
pasted_documents = Table(
    "pasted_documents",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("trip_id", Uuid, nullable=False),
    Column("kind", String(10), nullable=False),
    Column("text", Text, nullable=False),
)

# Read-only for the worker (SELECT grant): the dates a trip runs on.
trips = Table(
    "trips",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("start_date", Date),
    Column("end_date", Date),
)

# Read-only for the worker (SELECT grant): a receipt image or bank screenshot
# waiting for `read_receipt`. The image never leaves a step.
expense_evidence = Table(
    "expense_evidence",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("trip_id", Uuid, nullable=False),
    Column("data", LargeBinary, nullable=False),
    Column("media_type", String(20), nullable=False),
)

# Place catalog: SELECT, INSERT and UPDATE for the worker (no DELETE). Only the
# columns the OSM import writes are mapped; the rest keep their server defaults
# (`typical_visit_min` 60, `iconic` false...). Unique keys the import relies on:
# `places(osm_type, osm_id)`; `cities.slug` is the primary key.
cities = Table(
    "cities",
    metadata,
    Column("slug", String(64), primary_key=True),
    Column("name", String(100), nullable=False),
    Column("country", String(2), nullable=False),
    Column("timezone", String(64), nullable=False),
    Column("currency", String(3), nullable=False),
    Column("center_lat", Double, nullable=False),
    Column("center_lon", Double, nullable=False),
    Column("bbox_south", Double, nullable=False),
    Column("bbox_west", Double, nullable=False),
    Column("bbox_north", Double, nullable=False),
    Column("bbox_east", Double, nullable=False),
)

places = Table(
    "places",
    metadata,
    Column("id", Uuid, primary_key=True, server_default=func.gen_random_uuid()),
    Column("city_slug", String(64), nullable=False),
    Column("name", String(200), nullable=False),
    Column("category", String(32), nullable=False),
    Column("tags", ARRAY(String(32)), nullable=False),
    Column("lat", Double, nullable=False),
    Column("lon", Double, nullable=False),
    Column("osm_type", String(8)),
    Column("osm_id", BigInteger),
    # none_as_null: Python None is SQL NULL ("no hours"), not the JSON value null.
    Column("opening_hours", JSONB(none_as_null=True)),
    Column("hours_verified", Boolean, nullable=False),
    Column("wheelchair", Boolean),
    Column("indoor", Boolean),
    Column("cuisine", String(32)),
    Column("diet_tags", ARRAY(String(32)), nullable=False),
    Column("amenities", ARRAY(String(32)), nullable=False),
    Column("source", String(16), nullable=False),
)

# The inbox the frontend reads (backend#133): the worker may SELECT, INSERT and
# DELETE (no UPDATE). A trigger on INSERT wakes the live stream. Unique key:
# (user_sub, dedupe_key). `read_at` and `created_at` keep their server defaults
# on insert. The JSON variant lets tests run the purge on SQLite.
notifications = Table(
    "notifications",
    metadata,
    Column("id", Uuid, primary_key=True),
    Column("user_sub", String(255), nullable=False),
    Column("type", String(64), nullable=False),
    Column("trip_id", Uuid),
    Column("params", JSONB().with_variant(JSON(), "sqlite"), nullable=False),
    Column("actions", JSONB().with_variant(JSON(), "sqlite"), nullable=False),
    Column("dedupe_key", String(255)),
    Column("read_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# OSM fetch state, SELECT/INSERT/UPDATE for the worker (backend#187): the last
# fetch of a city (one row per city, it also starts the refresh period) and one
# reservation row per `fetch_place_candidates` workflow (the daily Overpass quota).
city_fetches = Table(
    "city_fetches",
    metadata,
    Column("city_slug", String(64), primary_key=True),
    Column("osm_relation_id", BigInteger, nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("stored", Integer, nullable=False),
)

city_fetch_attempts = Table(
    "city_fetch_attempts",
    metadata,
    Column("workflow_id", String(300), primary_key=True),
    Column("city_slug", String(64), nullable=False),
    Column("reserved_at", DateTime(timezone=True), nullable=False),
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
