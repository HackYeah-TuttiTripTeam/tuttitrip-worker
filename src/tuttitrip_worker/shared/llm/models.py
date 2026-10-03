"""Chat and decision models for Pydantic AI agents.

Durable agents (``DBOSDurability``) cannot carry a ``Model`` instance across
a DBOS step boundary; only model-id strings cross, and the model is rebuilt
inside the step. Agents therefore use the ids below and the :data:`catalog`'s
``ResolveModelId`` capability turns them into real models built from
settings, lazily, so the worker starts without API keys and tests never build
a real provider. The ids are the same as in the backend's catalog.

====================== ================================================
``tuttitrip:agent``    Qwen3.8-27B (thinking) on the GB10, else OpenRouter
``tuttitrip:chat``     Qwen3.8-27B chat (no thinking) on the GB10, else OpenRouter
``tuttitrip:decide``   basal (local decision model), else Qwen chat
``tuttitrip:decide-laya``   Laya (local decision model), else Qwen chat
``tuttitrip:decide-cloud``  JEV decision model through OpenRouter
``tuttitrip:openrouter``    OpenRouter only (``generate_trip_plan``)
``tuttitrip:local``         the local OpenAI-compatible endpoint only
====================== ================================================

Decision models hand a step they cannot answer (``UnfillableRoute``) or fail
on (API error) to the next model in a ``FallbackModel``.

Tests swap every model for a ``TestModel``/``FunctionModel`` with
:meth:`ModelCatalog.override`.
"""

import os
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, Final

from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models import Model, ModelResolutionContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.models.system_one import SystemOneModel
from pydantic_ai.profiles.decision import DecisionModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.system_one import SystemOneProvider

from tuttitrip_worker.contracts import LlmProvider
from tuttitrip_worker.shared.config.settings import LlmSettings, get_settings

MODEL_ID_PREFIX: Final = "tuttitrip:"

# Limit of pick-one options the basal/Laya API answers; more raises `UserError`
# before any request is sent.
DECISION_PROFILE: Final = DecisionModelProfile(decision_max_choice_options=10)

# Sent when a key is not configured, so building a model never needs a key; the
# request then fails with 401 (a `ModelAPIError`) and the fallback takes over.
UNSET_KEY: Final = "unset"


class ModelKey(StrEnum):
    """Model ids the catalog serves (the part after ``tuttitrip:``)."""

    AGENT = "agent"
    CHAT = "chat"
    DECIDE = "decide"
    DECIDE_LAYA = "decide-laya"
    DECIDE_CLOUD = "decide-cloud"
    OPENROUTER = "openrouter"
    LOCAL = "local"


def model_id(key: ModelKey | LlmProvider) -> str:
    """Model-id string an agent passes to ``agent.run(model=...)``.

    Args:
        key: Which catalog model to use (a job's ``LlmProvider`` also works).

    Returns:
        A string such as ``tuttitrip:agent``.
    """
    return f"{MODEL_ID_PREFIX}{key.value}"


def _openrouter_key(settings: LlmSettings) -> str:
    return (
        settings.openrouter_api_key.get_secret_value()
        or os.environ.get("OPENROUTER_API_KEY")
        or UNSET_KEY
    )


def _openrouter(settings: LlmSettings) -> Model:
    return OpenRouterModel(
        settings.openrouter_model,
        provider=OpenRouterProvider(api_key=_openrouter_key(settings)),
    )


def _gb10_qwen(name: str, settings: LlmSettings) -> Model:
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(
            base_url=settings.gb10_base_url,
            api_key=settings.gb10_api_key.get_secret_value() or UNSET_KEY,
        ),
    )


def _decision(name: str, base_url: str, api_key: str) -> Model:
    return SystemOneModel(
        name,
        provider=SystemOneProvider(base_url=base_url, api_key=api_key or UNSET_KEY),
        profile=DECISION_PROFILE,
    )


def build_model(key: ModelKey | LlmProvider, settings: LlmSettings) -> Model:
    """Build the real model (or fallback chain) for a catalog id from settings.

    Args:
        key: Which catalog model to build.
        settings: LLM settings (endpoints, keys, model names).

    Returns:
        A Pydantic AI model; network calls happen only when it is used.
    """
    gb10_key = settings.gb10_api_key.get_secret_value()
    match ModelKey(key.value):  # a StrEnum member of LlmProvider shares the value
        case ModelKey.OPENROUTER:
            return _openrouter(settings)
        case ModelKey.LOCAL:
            return OpenAIChatModel(
                settings.local_model,
                provider=OpenAIProvider(
                    base_url=settings.local_base_url,
                    api_key=settings.local_api_key.get_secret_value(),
                ),
            )
        case ModelKey.AGENT | ModelKey.CHAT:
            name = (
                settings.gb10_agent_model
                if key is ModelKey.AGENT
                else settings.gb10_chat_model
            )
            return FallbackModel(_gb10_qwen(name, settings), _openrouter(settings))
        case ModelKey.DECIDE | ModelKey.DECIDE_LAYA:
            decision = (
                _decision(settings.basal_model, settings.basal_base_url, gb10_key)
                if key is ModelKey.DECIDE
                else _decision(settings.laya_model, settings.laya_base_url, gb10_key)
            )
            return FallbackModel(
                decision, _gb10_qwen(settings.gb10_chat_model, settings)
            )
        case ModelKey.DECIDE_CLOUD:
            return SystemOneModel(
                settings.jev_model,
                provider=SystemOneProvider(
                    base_url=settings.openrouter_base_url,
                    api_key=_openrouter_key(settings),
                ),
            )


class ModelCatalog:
    """Resolves ``tuttitrip:<key>`` ids to cached models."""

    def __init__(self) -> None:
        self._models: dict[ModelKey, Model] = {}
        self._override: Model | None = None

    def get(self, key: ModelKey | LlmProvider) -> Model:
        """Return the model for a catalog id (or the test override).

        Args:
            key: Which catalog model to use.

        Returns:
            The cached model.
        """
        if self._override is not None:
            return self._override
        model_key = ModelKey(key.value)
        if model_key not in self._models:
            self._models[model_key] = build_model(model_key, get_settings().llm)
        return self._models[model_key]

    def resolve(self, _ctx: ModelResolutionContext[Any], model_id: str) -> Model | None:
        """``ResolveModelId`` hook: map our ids to models, ignore others.

        Args:
            _ctx: Resolution context (unused; models do not depend on deps).
            model_id: The id the agent run asked for.

        Returns:
            The model, or ``None`` to let Pydantic AI resolve foreign ids.
        """
        if not model_id.startswith(MODEL_ID_PREFIX):
            return None
        return self.get(ModelKey(model_id.removeprefix(MODEL_ID_PREFIX)))

    def capability(self) -> ResolveModelId[Any]:
        """Capability to attach to every agent that uses this catalog.

        Returns:
            A ``ResolveModelId`` capability bound to this catalog.
        """
        return ResolveModelId(self.resolve)

    @contextmanager
    def override(self, model: Model) -> Generator[None]:
        """Resolve every id to ``model`` (tests).

        Args:
            model: Replacement, e.g. ``TestModel()``.

        Yields:
            Nothing; the override ends with the ``with`` block.
        """
        previous, self._override = self._override, model
        try:
            yield
        finally:
            self._override = previous


catalog = ModelCatalog()
"""Process-wide catalog used by every agent."""
