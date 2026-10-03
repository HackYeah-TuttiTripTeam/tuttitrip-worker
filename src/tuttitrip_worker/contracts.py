"""Canonical worker <-> backend job contract.

This module is the single source of truth for everything the backend needs to
enqueue a job and read its outcome: the DBOS application name, workflow and
queue names, the input/output/event payload models and the contract version.
The backend keeps a mirror (``src/tuttitrip/shared/jobs/contracts.py`` in
``tuttitrip-backend``). Both repositories render the same document with
:func:`contract_json` and commit it as ``contracts/jobs.schema.json``; CI in
both repos diffs the two files (``deploy/CONVENTIONS.md`` in the backend,
section "Integracja z workerem").

Wire format:

* DBOS portable JSON serialization for inputs, outputs and events.
* Every workflow takes exactly one positional argument, a JSON object, and
  returns a JSON object. Each side validates it with its own Pydantic models.
* Every payload carries ``contract_version``. The worker rejects versions it
  does not support with :class:`ContractError` (``code`` + ``data``), which
  DBOS stores as a portable error (``name``, ``message``, ``code``, ``data``).

This module is pure: it imports only the standard library and Pydantic
(enforced by ``tests/architecture``).
"""

import json
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final, Literal, NamedTuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

APPLICATION_NAME: Final = "tuttitrip-worker"
"""DBOS application name: worker ``DBOSConfig["name"]`` and the backend
``DBOSClient(application_name=...)``. Queues and workflows are owned by it."""

CONTRACT_VERSION: Final = 1
"""Version the backend sends today (highest version this worker speaks)."""

SUPPORTED_CONTRACT_VERSIONS: Final = frozenset({1})
"""Versions this worker accepts. During a breaking change it holds old and new;
the heartbeat advertises ``min()`` and ``max()`` of it."""

PROGRESS_EVENT: Final = "progress"
"""Key of the progress event (``DBOS.set_event``) the backend reads."""


class Queue(StrEnum):
    """DBOS queues the worker listens on (all registered by the worker)."""

    DEFAULT = "default"
    """Cheap, non-LLM work: ping, embeddings, enrichment, plan recomputation."""
    LOCAL_LLM = "local_llm"
    """Jobs that use the local GPU model; very low concurrency."""
    OPENROUTER = "openrouter"
    """Jobs that call OpenRouter; rate limited."""


class Workflow(StrEnum):
    """Registered DBOS workflow names the backend may enqueue."""

    GENERATE_TRIP_PLAN = "generate_trip_plan"
    EMBED_TEXTS = "embed_texts"
    PING = "ping"


class LlmProvider(StrEnum):
    """Which model backend an LLM job uses (see :func:`queue_for`)."""

    OPENROUTER = "openrouter"
    LOCAL = "local"


ProviderName = Literal["openrouter", "local"]
"""Wire type of :class:`LlmProvider` (a Literal keeps the schema inline)."""


class ErrorCode(StrEnum):
    """Machine-readable codes of :class:`ContractError`."""

    UNSUPPORTED_CONTRACT_VERSION = "unsupported_contract_version"
    INVALID_PAYLOAD = "invalid_payload"


class ContractPayload(BaseModel):
    """Base for every input and output: carries the contract version.

    Unknown fields are ignored (not rejected) so that an additive change can
    reach one side before the other.
    """

    model_config = ConfigDict(frozen=True)

    contract_version: int = CONTRACT_VERSION


# --- generate_trip_plan -------------------------------------------------------


class GenerateTripPlanInput(ContractPayload):
    """Input of ``generate_trip_plan``: draft a plan with an LLM agent."""

    trip_id: UUID
    request: str = Field(min_length=1, max_length=4000)
    provider: ProviderName = "openrouter"


class GenerateTripPlanOutput(ContractPayload):
    """Output of ``generate_trip_plan`` (the full result goes to ``job_results``)."""

    destination: str
    days: int
    highlights: list[str]


# --- embed_texts -------------------------------------------------------------------


class EmbedTextsInput(ContractPayload):
    """Input of ``embed_texts``: embed short texts of one source row."""

    source_kind: str = Field(min_length=1, max_length=50)
    source_id: str = Field(min_length=1, max_length=200)
    texts: list[str] = Field(min_length=1, max_length=64)


class EmbedTextsOutput(ContractPayload):
    """Output of ``embed_texts`` (vectors are in the ``embeddings`` table)."""

    model: str
    dimensions: int
    stored: int


# --- ping ------------------------------------------------------------------------


class PingInput(ContractPayload):
    """Input of ``ping``: an echo used by post-deploy smoke tests (no LLM)."""

    message: str = Field(max_length=200)


class PingOutput(ContractPayload):
    """Output of ``ping``."""

    message: str
    worker_app_version: str


# --- events and errors ---------------------------------------------------------------


class Progress(BaseModel):
    """Value of the ``progress`` event, for the frontend's progress bar."""

    stage: str
    percent: int = Field(ge=0, le=100)


class ContractErrorData(BaseModel):
    """``data`` of the portable workflow error raised by :class:`ContractError`."""

    code: ErrorCode
    supported_versions: list[int]
    received_version: Any = None
    errors: list[dict[str, Any]] = Field(default_factory=list)


class ContractError(ValueError):
    """The payload does not match the contract; the workflow ends with it.

    DBOS stores it as a portable error: ``name="ContractError"``, ``message``,
    ``code`` (an :class:`ErrorCode` value) and ``data``
    (:class:`ContractErrorData`). The backend shows ``message`` in
    ``GET /jobs/{id}`` (``status=ERROR``).
    """

    def __init__(self, message: str, data: ContractErrorData) -> None:
        super().__init__(message)
        self.code: str = data.code.value
        self.data: dict[str, Any] = data.model_dump(mode="json")


def parse_input[T: ContractPayload](model: type[T], payload: object) -> T:
    """Validate a raw workflow argument against the contract.

    The version check comes first and reads the raw value, so a payload
    without ``contract_version`` is rejected even though the model has a
    default for it.

    Args:
        model: Expected input model.
        payload: The JSON object DBOS deserialized from the queue.

    Returns:
        The validated input.

    Raises:
        ContractError: Unsupported ``contract_version`` or invalid payload.
    """
    supported = sorted(SUPPORTED_CONTRACT_VERSIONS)
    received = payload.get("contract_version") if isinstance(payload, dict) else None
    if received not in SUPPORTED_CONTRACT_VERSIONS:
        data = ContractErrorData(
            code=ErrorCode.UNSUPPORTED_CONTRACT_VERSION,
            supported_versions=supported,
            received_version=received,
        )
        msg = (
            f"contract_version {received!r} is not supported by this worker "
            f"(supported: {supported})"
        )
        raise ContractError(msg, data)
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(
            include_url=False, include_context=False, include_input=False
        )
        data = ContractErrorData(
            code=ErrorCode.INVALID_PAYLOAD,
            supported_versions=supported,
            received_version=received,
            errors=[dict(error) for error in errors],
        )
        msg = f"invalid {model.__name__}: {exc}"
        raise ContractError(msg, data) from exc


# --- registry -------------------------------------------------------------------


class WorkflowSpec(NamedTuple):
    """Default queue and payload models of one workflow."""

    queue: Queue
    input: type[ContractPayload]
    output: type[ContractPayload]


WORKFLOWS: Final[Mapping[Workflow, WorkflowSpec]] = {
    # Enqueue on queue_for(provider): `openrouter` (default) or `local_llm`.
    Workflow.GENERATE_TRIP_PLAN: WorkflowSpec(
        Queue.OPENROUTER, GenerateTripPlanInput, GenerateTripPlanOutput
    ),
    Workflow.EMBED_TEXTS: WorkflowSpec(
        Queue.DEFAULT, EmbedTextsInput, EmbedTextsOutput
    ),
    Workflow.PING: WorkflowSpec(Queue.DEFAULT, PingInput, PingOutput),
}

EVENTS: Final[Mapping[str, type[BaseModel]]] = {PROGRESS_EVENT: Progress}

SCHEDULED_WORKFLOWS: Final[Mapping[str, str]] = {"heartbeat": "*/30 * * * * *"}
"""Internal scheduled workflows (name -> 6-field cron), never enqueued by the
backend. ``heartbeat`` upserts ``worker_heartbeats`` every 30 seconds."""


def queue_for(provider: LlmProvider | ProviderName) -> Queue:
    """Queue an LLM job must be enqueued on, given its provider.

    Args:
        provider: Model backend chosen for the job.

    Returns:
        ``local_llm`` for the local GPU model, ``openrouter`` otherwise.
    """
    return (
        Queue.LOCAL_LLM
        if LlmProvider(provider) is LlmProvider.LOCAL
        else Queue.OPENROUTER
    )


# --- rendered document (contracts/jobs.schema.json) ----------------------------

type JSON = dict[str, JSON] | list[JSON] | str | int | float | bool | None


def _strip_docs(node: JSON) -> JSON:
    """Drop ``title``/``description`` so docstrings never cause drift.

    Args:
        node: Any JSON value.

    Returns:
        The same value without documentation keys.
    """
    if isinstance(node, dict):
        return {
            key: _strip_docs(value)
            for key, value in node.items()
            if key not in {"title", "description"}
        }
    if isinstance(node, list):
        return [_strip_docs(value) for value in node]
    return node


def _schema(model: type[BaseModel]) -> JSON:
    return _strip_docs(model.model_json_schema())


def contract_document() -> dict[str, JSON]:
    """Language-neutral description of the contract (same shape as the backend).

    Returns:
        Version, names and JSON Schemas of all payloads and events.
    """
    queues: list[JSON] = [queue.value for queue in sorted(Queue)]
    workflows: dict[str, JSON] = {
        name.value: {
            "queue": spec.queue.value,
            "input": _schema(spec.input),
            "output": _schema(spec.output),
        }
        for name, spec in WORKFLOWS.items()
    }
    events: dict[str, JSON] = {name: _schema(model) for name, model in EVENTS.items()}
    return {
        "contract_version": CONTRACT_VERSION,
        "application_name": APPLICATION_NAME,
        "queues": queues,
        "workflows": workflows,
        "events": events,
    }


def contract_json() -> str:
    """Render the contract exactly as committed in ``contracts/jobs.schema.json``.

    Returns:
        Pretty-printed JSON with sorted keys and a trailing newline.
    """
    document = contract_document()
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
