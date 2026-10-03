"""Build idempotent embedding rows (pure)."""

from collections.abc import Sequence
from typing import Final
from uuid import UUID, uuid5

from tuttitrip_worker.embeddings.schemas import EmbeddingRow

EMBEDDING_NAMESPACE: Final = UUID("6f0d5c1e-3b8a-5d2e-9f47-2a1c7e9b4d10")
"""Fixed namespace of the deterministic row ids (never change it)."""


def embedding_id(*, source_kind: str, source_id: str, model: str, content: str) -> UUID:
    """Deterministic row id over source, model and text.

    The same text of the same source and model maps to the same row, so
    re-running a job is an upsert, not a duplicate.

    Args:
        source_kind: Kind of source row, e.g. ``place``.
        source_id: Backend id of the source row.
        model: Embedding model name.
        content: The embedded text.

    Returns:
        A UUIDv5 over all four values.
    """
    key = f"{source_kind}\x1f{source_id}\x1f{model}\x1f{content}"
    return uuid5(EMBEDDING_NAMESPACE, key)


def unique_texts(texts: Sequence[str]) -> list[str]:
    """Drop duplicate texts, keeping the first occurrence's order.

    Args:
        texts: Texts as sent by the backend.

    Returns:
        Texts without duplicates.
    """
    return list(dict.fromkeys(texts))


def build_rows(
    *,
    source_kind: str,
    source_id: str,
    model: str,
    texts: Sequence[str],
    vectors: Sequence[Sequence[float]],
) -> list[EmbeddingRow]:
    """Pair texts with their vectors.

    Args:
        source_kind: Kind of source row, e.g. ``place``.
        source_id: Backend id of the source row.
        model: Embedding model name.
        texts: Embedded texts.
        vectors: One vector per text, same order.

    Returns:
        Rows ready for the upsert.

    Raises:
        ValueError: ``texts`` and ``vectors`` differ in length.
    """
    if len(texts) != len(vectors):
        msg = f"{len(texts)} texts but {len(vectors)} vectors"
        raise ValueError(msg)
    return [
        EmbeddingRow(
            id=embedding_id(
                source_kind=source_kind, source_id=source_id, model=model, content=text
            ),
            source_kind=source_kind,
            source_id=source_id,
            content=text,
            model=model,
            embedding=list(vector),
        )
        for text, vector in zip(texts, vectors, strict=True)
    ]
