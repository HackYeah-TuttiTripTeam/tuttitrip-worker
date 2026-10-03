"""Text embeddings through an OpenAI-compatible endpoint (Ollama by default).

Embedding calls are network I/O, so callers run them inside a DBOS step.
Tests replace the model with ``TestEmbeddingModel`` via
:meth:`EmbedderCatalog.override`.
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager

from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingModel
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.providers.openai import OpenAIProvider

from tuttitrip_worker.shared.config.settings import LlmSettings, get_settings


def build_embedding_model(settings: LlmSettings) -> EmbeddingModel:
    """Build the real embedding model from settings.

    Args:
        settings: LLM settings (embedding endpoint, key and model).

    Returns:
        An embedding model; network calls happen only when it is used.
    """
    return OpenAIEmbeddingModel(
        settings.embedding_model,
        provider=OpenAIProvider(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key.get_secret_value(),
        ),
    )


class EmbedderCatalog:
    """Lazily built, process-wide embedder with a test override."""

    def __init__(self) -> None:
        self._embedder: Embedder | None = None
        self._override: Embedder | None = None
        self._override_name: str | None = None

    def get(self) -> Embedder:
        """Return the embedder (or the test override).

        Returns:
            The cached embedder.
        """
        if self._override is not None:
            return self._override
        if self._embedder is None:
            self._embedder = Embedder(build_embedding_model(get_settings().llm))
        return self._embedder

    @property
    def model_name(self) -> str:
        """Name stored next to each vector (part of the upsert key).

        Returns:
            The embedding model's name.
        """
        if self._override_name is not None:
            return self._override_name
        return get_settings().llm.embedding_model

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents (not search queries).

        Args:
            texts: Texts to embed, in order.

        Returns:
            One vector per text, in the same order.
        """
        result = await self.get().embed_documents(list(texts))
        return [list(vector) for vector in result.embeddings]

    @contextmanager
    def override(self, model: EmbeddingModel) -> Generator[None]:
        """Use ``model`` instead of the configured one (tests).

        Args:
            model: Replacement, e.g. ``TestEmbeddingModel(dimensions=768)``.

        Yields:
            Nothing; the override ends with the ``with`` block.
        """
        previous = self._override, self._override_name
        self._override, self._override_name = Embedder(model), model.model_name
        try:
            yield
        finally:
            self._override, self._override_name = previous


embedder = EmbedderCatalog()
"""Process-wide embedder used by embedding steps."""
