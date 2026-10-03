"""Internal payloads of the embeddings domain (pure: Pydantic only)."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict


class EmbeddingRow(BaseModel):
    """One row of the ``embeddings`` table."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    source_kind: str
    source_id: str
    content: str
    model: str
    embedding: list[float]


class EmbeddingBatch(BaseModel):
    """Vectors returned by the embedding step, with the model that made them."""

    model: str
    vectors: list[list[float]]
