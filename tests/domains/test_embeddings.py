"""Embeddings domain: pure row building, the workflow, the upsert SQL."""

from typing import Any

import pytest
from dbos import DBOSClient
from pydantic_ai.embeddings.test import TestEmbeddingModel
from sqlalchemy.dialects import postgresql

from tests.helpers import enqueue
from tuttitrip_worker.contracts import CONTRACT_VERSION, EmbedTextsOutput, Workflow
from tuttitrip_worker.embeddings import steps
from tuttitrip_worker.embeddings.logic.rows import (
    build_rows,
    embedding_id,
    unique_texts,
)
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.llm.embeddings import embedder

KEY = {"source_kind": "place", "source_id": "wawel", "model": "m"}


def test_embedding_id_is_deterministic_and_content_sensitive() -> None:
    first = embedding_id(**KEY, content="Zamek")
    assert first == embedding_id(**KEY, content="Zamek")
    assert first != embedding_id(**KEY, content="zamek")
    assert first != embedding_id(**{**KEY, "model": "other"}, content="Zamek")


def test_unique_texts_keeps_first_occurrence_order() -> None:
    assert unique_texts(["b", "a", "b", "c", "a"]) == ["b", "a", "c"]


def test_build_rows_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="2 texts but 1 vectors"):
        build_rows(**KEY, texts=["a", "b"], vectors=[[0.0]])


def test_upsert_is_keyed_by_the_deterministic_id() -> None:
    row = build_rows(**KEY, texts=["a"], vectors=[[0.5]])[0]
    statement = steps.build_upsert([row.model_dump()])
    sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "INSERT INTO embeddings" in sql
    assert "ON CONFLICT (id) DO UPDATE SET embedding = excluded.embedding" in sql


def test_embed_texts_embeds_unique_texts_and_upserts(
    client: DBOSClient, dbos: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    written: list[dict[str, Any]] = []

    async def fake_upsert(rows: list[dict[str, Any]]) -> int:
        written.extend(rows)
        return len(rows)

    monkeypatch.setattr(steps, "upsert_embeddings", fake_upsert)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "source_kind": "place",
        "source_id": "wawel",
        "texts": ["Zamek na wzgórzu", "Zamek na wzgórzu", "Smok wawelski"],
    }
    with embedder.override(TestEmbeddingModel("nomic-test", dimensions=4)):
        handle = enqueue(client, dbos, Workflow.EMBED_TEXTS, payload)
        output = EmbedTextsOutput.model_validate(handle.get_result())

    assert output == EmbedTextsOutput(model="nomic-test", dimensions=4, stored=2)
    assert [row["content"] for row in written] == ["Zamek na wzgórzu", "Smok wawelski"]
    expected_id = embedding_id(
        source_kind="place",
        source_id="wawel",
        model="nomic-test",
        content="Smok wawelski",
    )
    assert written[1]["id"] == str(expected_id)
    progress = client.get_event(handle.get_workflow_id(), "progress")
    assert progress == {"stage": "done", "percent": 100}
