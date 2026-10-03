"""Model and embedder construction from settings (no network)."""

from typing import Literal

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, models
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.decision import UnfillableRoute
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.models.test import TestModel

from tuttitrip_worker.contracts import LlmProvider
from tuttitrip_worker.shared.config.settings import LlmSettings
from tuttitrip_worker.shared.llm.models import (
    DECISION_PROFILE,
    ModelCatalog,
    ModelKey,
    build_model,
    model_id,
)


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


def legs(model: object) -> list[str]:
    assert isinstance(model, FallbackModel)
    return [leg.model_name for leg in model.models]


def test_ids_are_the_ones_of_the_backend_catalog() -> None:
    assert [model_id(key) for key in ModelKey] == [
        "tuttitrip:agent",
        "tuttitrip:chat",
        "tuttitrip:decide",
        "tuttitrip:decide-laya",
        "tuttitrip:decide-cloud",
        "tuttitrip:openrouter",
        "tuttitrip:local",
    ]


def test_agent_and_chat_are_qwen_with_an_openrouter_fallback() -> None:
    settings = LlmSettings(gb10_base_url="http://gb10/v1")
    agent = build_model(ModelKey.AGENT, settings)
    chat = build_model(ModelKey.CHAT, settings)
    assert legs(agent) == ["qwen3.8-27b", settings.openrouter_model]
    assert legs(chat) == ["qwen3.8-27b-chat", settings.openrouter_model]
    assert isinstance(agent, FallbackModel)
    assert isinstance(agent.models[0], OpenAIChatModel)
    assert agent.models[0].base_url == "http://gb10/v1/"
    assert isinstance(agent.models[1], OpenRouterModel)


def test_decide_is_basal_with_qwen_chat_behind_it() -> None:
    model = build_model(ModelKey.DECIDE, LlmSettings(basal_base_url="http://gb10/b/v1"))
    assert legs(model) == ["basal", "qwen3.8-27b-chat"]
    assert isinstance(model, FallbackModel)
    basal = model.models[0]
    assert isinstance(basal, SystemOneModel)
    assert basal.base_url == "http://gb10/b/v1"
    assert basal.profile.get("decision_max_choice_options") == 10  # type: ignore[attr-defined]
    assert DECISION_PROFILE["decision_max_choice_options"] == 10


def test_laya_and_jev_are_decision_models() -> None:
    laya = build_model(ModelKey.DECIDE_LAYA, LlmSettings())
    assert legs(laya) == ["laya", "qwen3.8-27b-chat"]
    jev = build_model(ModelKey.DECIDE_CLOUD, LlmSettings())
    assert isinstance(jev, SystemOneModel)
    assert jev.model_name == "typesafe/jev-1.13"
    assert jev.base_url == "https://openrouter.ai/api/v1"


def test_models_build_without_any_key() -> None:
    for key in ModelKey:
        assert build_model(key, LlmSettings()) is not None
    assert build_model(ModelKey.AGENT, LlmSettings()).model_name.startswith("fallback")


class Pick(BaseModel):
    choice: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]


def test_eleven_options_fail_before_any_request() -> None:
    # Nothing listens here: sending a request would fail with an API error.
    settings = LlmSettings(
        basal_base_url="http://127.0.0.1:9/v1", gb10_base_url="http://127.0.0.1:9/v1"
    )
    agent = Agent(build_model(ModelKey.DECIDE, settings), output_type=Pick)
    with models.override_allow_model_requests(True), pytest.raises(UserError):  # ruff: ignore[boolean-positional-value-in-call]
        agent.run_sync("wybierz")


def test_unfillable_route_escalates_to_the_language_model() -> None:
    basal_name = "basal"

    def decision(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        raise UnfillableRoute(basal_name, "Pick", 0.9)

    def qwen(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("odpowiada Qwen")])

    chain = FallbackModel(FunctionModel(decision), FunctionModel(qwen))
    assert Agent(chain).run_sync("pytanie").output == "odpowiada Qwen"
