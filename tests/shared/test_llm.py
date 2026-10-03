"""Model and embedder construction from settings (no network)."""

import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.test import TestModel

from tuttitrip_worker.contracts import LlmProvider
from tuttitrip_worker.shared.config.settings import LlmSettings
from tuttitrip_worker.shared.llm.models import ModelCatalog, build_model, model_id


def test_openrouter_model_uses_settings_key() -> None:
    settings = LlmSettings(openrouter_api_key="sk-test", openrouter_model="a/b")
    model = build_model(LlmProvider.OPENROUTER, settings)
    assert isinstance(model, OpenRouterModel)
    assert model.model_name == "a/b"


def test_openrouter_falls_back_to_the_standard_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
    model = build_model(LlmProvider.OPENROUTER, LlmSettings())
    assert isinstance(model, OpenRouterModel)


def test_local_model_targets_the_configured_endpoint() -> None:
    settings = LlmSettings(local_base_url="http://gpu:30000/v1", local_model="qwen")
    model = build_model(LlmProvider.LOCAL, settings)
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "qwen"
    assert model.base_url == "http://gpu:30000/v1/"


def test_catalog_resolves_only_our_ids() -> None:
    catalog = ModelCatalog()
    test_model = TestModel()
    with catalog.override(test_model):
        assert catalog.get(LlmProvider.LOCAL) is test_model
    assert model_id(LlmProvider.LOCAL) == "tuttitrip:local"
