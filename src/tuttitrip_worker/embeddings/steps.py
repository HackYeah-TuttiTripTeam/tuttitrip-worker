"""I/O steps of the embeddings domain (model calls and database writes)."""

from collections.abc import Sequence
from typing import Any

from dbos import DBOS
from sqlalchemy.dialects.postgresql import Insert, insert

from tuttitrip_worker.embeddings.schemas import EmbeddingBatch, EmbeddingRow
from tuttitrip_worker.shared.config.settings import get_settings
from tuttitrip_worker.shared.db.engine import transaction
from tuttitrip_worker.shared.db.tables import embeddings
from tuttitrip_worker.shared.llm.embeddings import embedder


@DBOS.step(
    retries_allowed=True,
    max_attempts=get_settings().dbos.embed_max_attempts,
    interval_seconds=get_settings().dbos.embed_retry_interval_seconds,
)
async def embed(texts: list[str]) -> dict[str, Any]:
    """Call the embedding model (retried on failure).

    Args:
        texts: Texts to embed.

    Returns:
        ``EmbeddingBatch`` as a JSON object (step outputs are checkpointed).
    """
    vectors = await embedder.embed_documents(texts)
    return EmbeddingBatch(model=embedder.model_name, vectors=vectors).model_dump()


def build_upsert(values: Sequence[dict[str, Any]]) -> Insert:
    """``INSERT ... ON CONFLICT (id) DO UPDATE`` for embedding rows.

    Args:
        values: Column values per row (``EmbeddingRow`` dumps).

    Returns:
        The statement (not executed).
    """
    statement = insert(embeddings).values(list(values))
    return statement.on_conflict_do_update(
        index_elements=[embeddings.c.id],
        set_={"embedding": statement.excluded.embedding},
    )


@DBOS.step(retries_allowed=True, max_attempts=get_settings().dbos.step_max_attempts)
async def upsert_embeddings(rows: Sequence[dict[str, Any]]) -> int:
    """Insert or refresh rows by their deterministic id.

    Args:
        rows: ``EmbeddingRow`` objects as JSON.

    Returns:
        Number of rows written.
    """
    if not rows:
        return 0
    values = [EmbeddingRow.model_validate(row).model_dump() for row in rows]
    async with transaction() as connection:
        await connection.execute(build_upsert(values))
    return len(values)
