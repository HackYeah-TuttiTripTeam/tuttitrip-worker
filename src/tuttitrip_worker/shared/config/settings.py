"""Typed worker settings (pydantic-settings).

Fields map to environment variables with the ``TUTTITRIP_`` prefix; nested
models use ``__`` (``TUTTITRIP_LLM__LOCAL_MODEL``). Three fields read the names
the backend deploy writes into ``~/tuttitrip/envs/<env>.worker.env``
(``deploy/CONVENTIONS.md`` in the backend): ``TUTTITRIP_WORKER_DATABASE_URL``,
``DBOS_SYSTEM_DATABASE_URL`` and ``DBOS__APPVERSION``.
``.env.example`` must list exactly these variables (a test enforces it).
"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_PREFIX = "TUTTITRIP_"
ENV_NESTED_DELIMITER = "__"
LOCAL_DATABASE_URL = "postgresql://tuttitrip:tuttitrip@localhost:5432/tuttitrip"


class DbosSettings(BaseModel):
    """DBOS runtime options."""

    # Seconds `docker stop` gives running workflows before we exit; keep it
    # below the container stop timeout (the deploy uses --stop-timeout 40).
    shutdown_timeout_sec: int = Field(default=30, ge=0, le=600)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class LlmSettings(BaseModel):
    """Model providers for Pydantic AI agents and embeddings."""

    # OpenRouter. Empty key = fall back to the standard OPENROUTER_API_KEY.
    openrouter_api_key: SecretStr = SecretStr("")
    openrouter_model: str = "google/gemini-3.8-flash"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # JEV decision model (`tuttitrip:decide-cloud`), served through OpenRouter.
    jev_model: str = "typesafe/jev-1.13"
    # GB10 (dellpromaxgb10) behind LiteLLM: Qwen (OpenAI-compatible) and the
    # decision models basal and Laya. Empty key = requests fail with 401 and
    # the fallback (OpenRouter or Qwen) answers.
    gb10_base_url: str = "https://llm.gburek.app/v1"
    gb10_api_key: SecretStr = SecretStr("")
    gb10_agent_model: str = "qwen3.8-27b"
    gb10_chat_model: str = "qwen3.8-27b-chat"
    basal_base_url: str = "https://llm.gburek.app/basal/v1"
    basal_model: str = "basal"
    laya_base_url: str = "https://llm.gburek.app/laya/v1"
    laya_model: str = "laya"
    # Local OpenAI-compatible chat endpoint (SGLang/vLLM/Ollama on the GPU host).
    local_base_url: str = "http://localhost:30000/v1"
    local_api_key: SecretStr = SecretStr("local")
    local_model: str = "qwen3.8-27b"
    # OpenAI-compatible embeddings endpoint (Ollama with nomic-embed-text).
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: SecretStr = SecretStr("ollama")
    embedding_model: str = "nomic-embed-text"
    embedding_dimensions: int = Field(default=768, ge=1, le=16000)


class Settings(BaseSettings):
    """Root settings object for the worker."""

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter=ENV_NESTED_DELIMITER,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "local"
    # Application database as the restricted role `tuttitrip_worker` (on the
    # host); locally the owner role of the backend's compose database.
    worker_database_url: SecretStr = SecretStr(LOCAL_DATABASE_URL)
    # DBOS system database (schema `dbos`, migrated by the backend); empty =
    # the application database above.
    dbos_system_database_url: SecretStr = Field(
        default=SecretStr(""), validation_alias="DBOS_SYSTEM_DATABASE_URL"
    )
    # Constant per environment (`<env>`); the backend enqueues with the same
    # value (`app_version`), so its jobs always match a running worker.
    application_version: str = Field(
        default="local", validation_alias="DBOS__APPVERSION"
    )
    dbos: DbosSettings = Field(default_factory=DbosSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)

    def system_database_url(self) -> str:
        """DBOS system database URL (defaults to the application database).

        Returns:
            The URL DBOS uses for its ``dbos`` schema.
        """
        explicit = self.dbos_system_database_url.get_secret_value()
        return explicit or self.worker_database_url.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process.

    Returns:
        The cached worker settings.
    """
    return Settings()
