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


KEYS = LlmSettings(gb10_api_key="k-gb10", openrouter_api_key="k-or")
DECISION_KEYS = [ModelKey.DECIDE, ModelKey.DECIDE_LAYA, ModelKey.DECIDE_CLOUD]


def legs(model: object) -> list[str]:
    assert isinstance(model, FallbackModel)
    return [leg.model_name for leg in model.models]


def test_ids_are_the_ones_of_the_backend_catalog() -> None:
    assert [model_id(key) for key in ModelKey] == [
        "tuttitrip:agent",
        "tuttitrip:chat",
        "tuttitrip:vision",
        "tuttitrip:decide",
        "tuttitrip:decide-laya",
        "tuttitrip:decide-cloud",
        "tuttitrip:openrouter",
        "tuttitrip:local",
    ]


def test_agent_and_chat_are_qwen_with_an_openrouter_fallback() -> None:
    settings = KEYS.model_copy(update={"gb10_base_url": "http://gb10/v1"})
    agent = build_model(ModelKey.AGENT, settings)
    assert legs(agent) == ["qwen3.8-27b", settings.openrouter_model]
    assert legs(build_model(ModelKey.CHAT, settings)) == [
        "qwen3.8-27b-chat",
        settings.openrouter_model,
    ]
    assert isinstance(agent, FallbackModel)
    assert isinstance(agent.models[0], OpenAIChatModel)
    assert agent.models[0].base_url == "http://gb10/v1/"
    assert isinstance(agent.models[1], OpenRouterModel)


def test_decide_and_laya_escalate_to_qwen_chat() -> None:
    settings = KEYS.model_copy(update={"basal_base_url": "http://gb10/b/v1"})
    basal = build_model(ModelKey.DECIDE, settings)
    assert legs(basal) == ["basal", "qwen3.8-27b-chat"]
    assert isinstance(basal, FallbackModel)
    assert isinstance(basal.models[0], SystemOneModel)
    assert basal.models[0].base_url == "http://gb10/b/v1"
    assert legs(build_model(ModelKey.DECIDE_LAYA, KEYS)) == ["laya", "qwen3.8-27b-chat"]


def test_jev_goes_through_the_openrouter_base_url() -> None:
    jev = build_model(ModelKey.DECIDE_CLOUD, KEYS)
    assert isinstance(jev, SystemOneModel)
    assert jev.model_name == "typesafe/jev-1.13"
    assert jev.base_url == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize("key", DECISION_KEYS)
def test_every_decision_model_has_the_ten_option_limit(key: ModelKey) -> None:
    model = build_model(key, KEYS)
    decision = model.models[0] if isinstance(model, FallbackModel) else model
    assert isinstance(decision, SystemOneModel)
    assert decision.profile.get("decision_max_choice_options") == 10  # type: ignore[attr-defined]
    assert DECISION_PROFILE["decision_max_choice_options"] == 10


def test_links_without_a_key_are_skipped() -> None:
    only_or = LlmSettings(openrouter_api_key="k-or")
    assert isinstance(build_model(ModelKey.AGENT, only_or), OpenRouterModel)
    assert isinstance(build_model(ModelKey.CHAT, only_or), OpenRouterModel)
    only_gb10 = LlmSettings(gb10_api_key="k-gb10")
    assert isinstance(build_model(ModelKey.AGENT, only_gb10), OpenAIChatModel)
    assert legs(build_model(ModelKey.DECIDE, only_gb10)) == [
        "basal",
        "qwen3.8-27b-chat",
    ]


@pytest.mark.parametrize("key", [key for key in ModelKey if key is not ModelKey.LOCAL])
def test_a_chain_without_any_key_raises_a_user_error(key: ModelKey) -> None:
    with pytest.raises(UserError, match=model_id(key)):
        build_model(key, LlmSettings())


def test_local_needs_no_key() -> None:
    assert isinstance(build_model(ModelKey.LOCAL, LlmSettings()), OpenAIChatModel)


class Pick(BaseModel):
    choice: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"]


@pytest.mark.parametrize("key", DECISION_KEYS)
def test_eleven_options_fail_before_any_request(key: ModelKey) -> None:
    # Nothing listens here: sending a request would fail with an API error.
    dead = "http://127.0.0.1:9/v1"
    settings = KEYS.model_copy(
        update={
            "basal_base_url": dead,
            "laya_base_url": dead,
            "gb10_base_url": dead,
            "openrouter_base_url": dead,
        }
    )
    agent = Agent(build_model(key, settings), output_type=Pick)
    # The limit is checked inside the decision model, after the allow-requests
    # gate, so the gate is opened; the UserError comes before any HTTP call.
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


def test_vision_is_the_gb10_alone_and_never_falls_back_to_a_cloud() -> None:
    model = build_model(ModelKey.VISION, KEYS)  # both keys are set
    assert isinstance(model, OpenAIChatModel)
    assert not isinstance(model, FallbackModel)
    assert model.model_name == "qwen3.8-27b-chat"
    with pytest.raises(UserError, match="GB10"):
        build_model(ModelKey.VISION, LlmSettings(openrouter_api_key="k-or"))
