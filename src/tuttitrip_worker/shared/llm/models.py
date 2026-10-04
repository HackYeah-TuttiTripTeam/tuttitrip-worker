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

from openai import AsyncOpenAI
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.exceptions import UserError
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

# Pick-one limit of every decision model (basal, Laya, JEV); more options raise
# `UserError` before any request is sent.
DECISION_PROFILE: Final = DecisionModelProfile(decision_max_choice_options=10)


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
        or ""
    )


def build_openrouter(name: str, settings: LlmSettings) -> Model | None:
    """Build an OpenRouter model by its slug.

    Args:
        name: OpenRouter model slug, for example ``google/gemini-3.8-flash``.
        settings: LLM settings (endpoint and key).

    Returns:
        The model, or ``None`` when no OpenRouter key is configured.
    """
    key = _openrouter_key(settings)
    if not key:
        return None
    # OpenRouterProvider has no base_url argument; a client carries it.
    client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=key)
    return OpenRouterModel(name, provider=OpenRouterProvider(openai_client=client))


def _openrouter(settings: LlmSettings) -> Model | None:
    return build_openrouter(settings.openrouter_model, settings)


def _gb10_qwen(name: str, settings: LlmSettings) -> Model | None:
    key = settings.gb10_api_key.get_secret_value()
    if not key:
        return None
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(base_url=settings.gb10_base_url, api_key=key),
    )


def _decision(name: str, base_url: str, settings: LlmSettings) -> Model | None:
    key = settings.gb10_api_key.get_secret_value()
    if not key:
        return None
    return SystemOneModel(
        name,
        provider=SystemOneProvider(base_url=base_url, api_key=key),
        profile=DECISION_PROFILE,
    )


def _chain(key: ModelKey, *links: Model | None) -> Model:
    """Join the links that have a key into a ``FallbackModel``.

    Args:
        key: Catalog id, for the error message.
        *links: Models in order of preference; ``None`` = no key configured.

    Returns:
        The only link, or a ``FallbackModel`` of all of them.

    Raises:
        UserError: No link has a key.
    """
    models = [link for link in links if link is not None]
    if not models:
        msg = (
            f"No API key is configured for {model_id(key)}: set "
            "TUTTITRIP_LLM__GB10_API_KEY and/or OPENROUTER_API_KEY."
        )
        raise UserError(msg)
    return models[0] if len(models) == 1 else FallbackModel(*models)


def build_model(key: ModelKey | LlmProvider, settings: LlmSettings) -> Model:
    """Build the real model (or fallback chain) for a catalog id from settings.

    Links whose API key is missing are left out of the chain.

    Args:
        key: Which catalog model to build.
        settings: LLM settings (endpoints, keys, model names).

    Returns:
        A Pydantic AI model; network calls happen only when it is used.
    """
    model_key = ModelKey(key.value)  # LlmProvider members share the values
    qwen_chat = _gb10_qwen(settings.gb10_chat_model, settings)
    match model_key:
        case ModelKey.OPENROUTER:
            return _chain(model_key, _openrouter(settings))
        case ModelKey.LOCAL:
            return OpenAIChatModel(
                settings.local_model,
                provider=OpenAIProvider(
                    base_url=settings.local_base_url,
                    api_key=settings.local_api_key.get_secret_value(),
                ),
            )
        case ModelKey.AGENT:
            qwen = _gb10_qwen(settings.gb10_agent_model, settings)
            return _chain(model_key, qwen, _openrouter(settings))
        case ModelKey.CHAT:
            return _chain(model_key, qwen_chat, _openrouter(settings))
        case ModelKey.DECIDE | ModelKey.DECIDE_LAYA:
            name, url = {
                ModelKey.DECIDE: (settings.basal_model, settings.basal_base_url),
                ModelKey.DECIDE_LAYA: (settings.laya_model, settings.laya_base_url),
            }[model_key]
            return _chain(model_key, _decision(name, url, settings), qwen_chat)
        case ModelKey.DECIDE_CLOUD:
            key_ = _openrouter_key(settings)
            jev = (
                SystemOneModel(
                    settings.jev_model,
                    provider=SystemOneProvider(
                        base_url=settings.openrouter_base_url, api_key=key_
                    ),
                    profile=DECISION_PROFILE,
                )
                if key_
                else None
            )
            return _chain(model_key, jev)


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
