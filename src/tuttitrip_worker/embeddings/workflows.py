"""Embedding workflows: texts to vectors, upserted into pgvector."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.contracts import (
    EmbedTextsInput,
    EmbedTextsOutput,
    Workflow,
    parse_input,
)
from tuttitrip_worker.embeddings import steps
from tuttitrip_worker.embeddings.constants import PROGRESS_EMBEDDING, PROGRESS_STORING
from tuttitrip_worker.embeddings.logic.rows import build_rows, unique_texts
from tuttitrip_worker.embeddings.schemas import EmbeddingBatch
from tuttitrip_worker.shared.dbos.constants import PROGRESS_DONE
from tuttitrip_worker.shared.dbos.runtime import PORTABLE, report_progress


@DBOS.workflow(name=Workflow.EMBED_TEXTS.value, serialization_type=PORTABLE)
async def embed_texts(payload: dict[str, Any]) -> dict[str, Any]:
    """Embed a source row's texts and upsert them (safe to re-run).

    Args:
        payload: JSON object matching ``EmbedTextsInput``.

    Returns:
        JSON object matching ``EmbedTextsOutput``.
    """
    request = parse_input(EmbedTextsInput, payload)
    texts = unique_texts(request.texts)
    await report_progress(*PROGRESS_EMBEDDING)
    batch = EmbeddingBatch.model_validate(await steps.embed(texts))
    await report_progress(*PROGRESS_STORING)
    rows = build_rows(
        source_kind=request.source_kind,
        source_id=request.source_id,
        model=batch.model,
        texts=texts,
        vectors=batch.vectors,
    )
    stored = await steps.upsert_embeddings(
        [row.model_dump(mode="json") for row in rows]
    )
    await report_progress(*PROGRESS_DONE)
    dimensions = len(batch.vectors[0]) if batch.vectors else 0
    output = EmbedTextsOutput(model=batch.model, dimensions=dimensions, stored=stored)
    return output.model_dump(mode="json")
