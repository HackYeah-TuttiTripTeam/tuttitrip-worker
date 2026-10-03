"""Chat models for Pydantic AI agents: OpenRouter and the local GPU endpoint.

Durable agents (``DBOSDurability``) cannot carry a ``Model`` instance across
a DBOS step boundary; only model-id strings cross, and the model is rebuilt
inside the step. Agents therefore use the ids below (``tuttitrip:openrouter``
and ``tuttitrip:local``) and the :data:`catalog`'s ``ResolveModelId``
capability turns them into real models built from settings, lazily, so the
worker starts without API keys and tests never build a real provider.

Tests swap every model for a ``TestModel``/``FunctionModel`` with
:meth:`ModelCatalog.override`.
"""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, Final

from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models import Model, ModelResolutionContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from tuttitrip_worker.contracts import LlmProvider
from tuttitrip_worker.shared.config.settings import LlmSettings, get_settings

MODEL_ID_PREFIX: Final = "tuttitrip:"


def model_id(provider: LlmProvider) -> str:
    """Model-id string an agent passes to ``agent.run(model=...)``.

    Args:
        provider: Which model backend to use.

    Returns:
        A string such as ``tuttitrip:openrouter``.
    """
    return f"{MODEL_ID_PREFIX}{provider.value}"


def build_model(provider: LlmProvider, settings: LlmSettings) -> Model:
    """Build the real chat model for a provider from settings.

    Args:
        provider: Which model backend to build.
        settings: LLM settings (endpoints, keys, model names).

    Returns:
        A Pydantic AI model; network calls happen only when it is used.
    """
    if provider is LlmProvider.OPENROUTER:
        key = settings.openrouter_api_key.get_secret_value() or None
        return OpenRouterModel(
            settings.openrouter_model, provider=OpenRouterProvider(api_key=key)
        )
    return OpenAIChatModel(
        settings.local_model,
        provider=OpenAIProvider(
            base_url=settings.local_base_url,
            api_key=settings.local_api_key.get_secret_value(),
        ),
    )


class ModelCatalog:
    """Resolves ``tuttitrip:<provider>`` ids to cached models."""

    def __init__(self) -> None:
        self._models: dict[LlmProvider, Model] = {}
        self._override: Model | None = None

    def get(self, provider: LlmProvider) -> Model:
        """Return the model for a provider (or the test override).

        Args:
            provider: Which model backend to use.

        Returns:
            The cached model.
        """
        if self._override is not None:
            return self._override
        if provider not in self._models:
            self._models[provider] = build_model(provider, get_settings().llm)
        return self._models[provider]

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
        return self.get(LlmProvider(model_id.removeprefix(MODEL_ID_PREFIX)))

    def capability(self) -> ResolveModelId[Any]:
        """Capability to attach to every agent that uses this catalog.

        Returns:
            A ``ResolveModelId`` capability bound to this catalog.
        """
        return ResolveModelId(self.resolve)

    @contextmanager
    def override(self, model: Model) -> Generator[None]:
        """Resolve every provider to ``model`` (tests).

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
