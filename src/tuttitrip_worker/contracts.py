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

Compatible changes since version 1 (no version bump, the JSON schema is
unchanged or only grows): ``ErrorCode.DOCUMENT_NOT_FOUND`` (code of the error
of a workflow whose pasted document does not exist; the backend maps it in
tuttitrip-backend#142) and ``ErrorCode.MODEL_OUTPUT_INVALID`` (the model never
produced a valid structured answer, for example ``parse_pasted_plan`` after the
last retry; the backend maps it like the others), ``ErrorCode.CITY_NOT_FOUND``
(``fetch_place_candidates`` cannot find the city in Nominatim, or ``city_slug``
names a city nobody has fetched yet) and ``ErrorCode.RATE_LIMITED`` (the daily
budget of Overpass queries is spent; try again tomorrow) and
``ErrorCode.SLUG_CONFLICT`` (the slug already belongs to another OSM city).

This module is pure: it imports only the standard library and Pydantic
(enforced by ``tests/architecture``).
"""

import json
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Final, Literal, NamedTuple, Self
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

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
    PARSE_PASTED_PLAN = "parse_pasted_plan"
    EXTRACT_OFFER_EVIDENCE = "extract_offer_evidence"
    FETCH_PLACE_CANDIDATES = "fetch_place_candidates"
    WRITE_JUSTIFICATIONS = "write_justifications"
    PARSE_EXPENSE_TEXT = "parse_expense_text"
    READ_RECEIPT = "read_receipt"


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
    NOT_IMPLEMENTED = "not_implemented"
    DOCUMENT_NOT_FOUND = "document_not_found"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    CITY_NOT_FOUND = "city_not_found"
    RATE_LIMITED = "rate_limited"
    SLUG_CONFLICT = "slug_conflict"


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


# --- notifications written by the worker -----------------------------------------

NotificationType = Literal["plan_ready"]
"""Types of notification the worker may insert (the backend owns the list,
``NotificationType`` in ``tuttitrip.notifications.schemas``; a test pins the
names). Not part of ``jobs.schema.json``: notifications are rows, not jobs."""

NotificationActionCode = Literal["open_trip", "open_people", "open_plan"]
"""Buttons a worker notification may carry (a subset of the backend's codes)."""


class NotificationDraft(BaseModel):
    """One notification to insert for a user (the recipient is passed apart).

    ``params`` are small strings for the text (names, ids); no tokens and
    nothing confidential, they reach the browser. ``dedupe_key`` names the thing
    and its version or day (``plan_ready:<trip>:<workflow>``), so a retry or a
    repeated job inserts nothing new.
    """

    model_config = ConfigDict(frozen=True)

    type: NotificationType
    trip_id: UUID | None = None
    params: dict[str, str] = Field(default_factory=dict, max_length=20)
    actions: list[NotificationActionCode] = Field(default_factory=list, max_length=4)
    dedupe_key: str = Field(min_length=1, max_length=255)


# --- shared helpers -----------------------------------------------------------------

SLUG_PATTERN: Final = r"^[a-z0-9]+(-[a-z0-9]+)*$"
"""City slug: lowercase ASCII words joined by ``-``. Built from a free-text city
name by lowercasing, removing diacritics (``ł`` becomes ``l``), turning every
run of other characters into one ``-`` and trimming ``-`` at both ends
(``"Gdańsk, Polska"`` becomes ``"gdansk-polska"``). Backend and worker use this
one rule."""

Locale = Literal["pl", "en"]

# --- parse_pasted_plan -------------------------------------------------------------

TIME_PATTERN: Final = r"^([01][0-9]|2[0-3]):[0-5][0-9]$"
"""24-hour ``HH:MM``, zero padded."""

TransportMode = Literal["walk", "public_transport", "car", "taxi", "bike", "other"]


def _normalize_time(value: object) -> object:
    """Zero-pad ``H:MM`` to ``HH:MM`` (``9:00`` becomes ``09:00``).

    Args:
        value: Raw value; anything but a string is returned unchanged.

    Returns:
        The padded time, or the input for the pattern check to judge.
    """
    if isinstance(value, str) and re.fullmatch(r"[0-9]:[0-5][0-9]", value.strip()):
        return f"0{value.strip()}"
    return value


class ParsedPlanItem(BaseModel):
    """One item read from a pasted plan; ``quote`` is verbatim from the text.

    Times and amounts are claims of the checked plan, not catalog data.
    ``start_time`` and ``end_time`` are ``HH:MM``; ``9:00`` is normalized to
    ``09:00``. ``amount_minor`` is in minor units (grosze, cents) and is a
    price **per person** (the total for a group is the linter's business);
    when the text gives only a group total, the parser leaves it ``None``.
    ``day`` is ``None`` when the text does not say which day (the item stays
    in text order, see ``index``).
    """

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    day: int | None = Field(default=None, ge=1, le=60)
    start_time: str | None = Field(default=None, pattern=TIME_PATTERN)
    end_time: str | None = Field(default=None, pattern=TIME_PATTERN)
    place_name: str = Field(min_length=1, max_length=200)
    address: str | None = Field(default=None, max_length=300)
    city: str | None = Field(default=None, max_length=100)
    amount_minor: int | None = Field(default=None, ge=0)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    transport: TransportMode | None = None
    quote: str = Field(min_length=1, max_length=1000)

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _pad_time(cls, value: object) -> object:
        return _normalize_time(value)


class MatchCandidate(BaseModel):
    """A catalog place proposed for a pasted item (pure code ranks them)."""

    model_config = ConfigDict(frozen=True)

    place_id: str = Field(min_length=1, max_length=100)
    name: str = Field(max_length=200)
    address: str | None = Field(default=None, max_length=300)
    category: str | None = Field(default=None, max_length=100)
    score: float = Field(ge=0, le=1)


class PlaceMatch(BaseModel):
    """Catalog match of one parsed item.

    ``status``: ``matched`` (``place_id`` set, confident), ``needs_confirmation``
    (low or unreported confidence; ``place_id`` holds the model's pick and the
    host confirms it or picks from ``candidates``) or ``unrecognized``
    (``place_id`` is ``None``). ``ParsePastedPlanOutput.matches`` has one entry
    per item.

    ``confidence`` depends on who decided. Model pick or "none of these": the
    decision model's margin (0..1, not a probability), ``None`` when a
    language-model fallback answered. No model (switched off, down, or no
    candidates): the name similarity of the best candidate for ``matched``;
    for ``unrecognized`` it is ``None``, so a missing value there means "no
    model said so", never "sure".
    """

    model_config = ConfigDict(frozen=True)

    item_index: int = Field(ge=0)
    status: Literal["matched", "needs_confirmation", "unrecognized"]
    place_id: str | None = Field(default=None, max_length=100)
    confidence: float | None = Field(default=None, ge=0, le=1)
    candidates: list[MatchCandidate] = Field(default_factory=list, max_length=9)


class UnreadItem(BaseModel):
    """Text the parser could not turn into a valid item (no verbatim quote)."""

    model_config = ConfigDict(frozen=True)

    quote: str = Field(max_length=1000)
    reason: Literal["quote_not_in_text", "invalid_item"]


class ParsePastedPlanInput(ContractPayload):
    """Input of ``parse_pasted_plan``; the text is read from ``pasted_documents``."""

    trip_id: UUID
    document_id: UUID
    city_slug: str = Field(pattern=SLUG_PATTERN, max_length=100)
    provider: ProviderName = "openrouter"


class ParsePastedPlanOutput(ContractPayload):
    """Output of ``parse_pasted_plan`` (also kept in ``job_results``)."""

    items: list[ParsedPlanItem] = Field(max_length=300)
    unread: list[UnreadItem] = Field(default_factory=list, max_length=300)
    matches: list[PlaceMatch] = Field(default_factory=list, max_length=300)


# --- extract_offer_evidence ----------------------------------------------------------

OfferVerdict = Literal["present", "absent", "not_applicable"]
"""What a quote says about a requirement. No quote means unconfirmed, which is
decided by the backend, never by the worker."""


class EvidenceQuote(BaseModel):
    """One verbatim quote of the offer and what it says about the requirement.

    ``verdict`` and ``confidence`` are ``None`` when the judge (decision model)
    was unavailable; ``confidence`` is also ``None`` when a language-model
    fallback answered. Confidence is the decision model's margin from its
    threshold scaled to 0..1, not a probability.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1, max_length=1000)
    verdict: OfferVerdict | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)


class RequirementEvidence(BaseModel):
    """Quotes of the offer for one requirement, each with its own assessment.

    ``quotes == []`` means the offer is silent about it (the backend calls it
    unconfirmed). Quotes of one key may disagree (``present`` and ``absent``);
    the worker does not aggregate, the backend shows such a key as conflicting.
    """

    model_config = ConfigDict(frozen=True)

    requirement_key: str = Field(min_length=1, max_length=100)
    quotes: list[EvidenceQuote] = Field(default_factory=list, max_length=10)


class RequirementLabel(BaseModel):
    """A requirement key with the human label the model reads (backend#67)."""

    model_config = ConfigDict(frozen=True)

    key: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=200)


class ExtractOfferEvidenceInput(ContractPayload):
    """Input of ``extract_offer_evidence``; the offer is in ``pasted_documents``.

    ``requirement_keys`` must be unique. ``requirements`` optionally gives a
    label for keys (each must also be in ``requirement_keys``).
    """

    trip_id: UUID
    document_id: UUID
    requirement_keys: list[str] = Field(min_length=1, max_length=50)
    requirements: list[RequirementLabel] | None = Field(default=None, max_length=50)
    provider: ProviderName = "openrouter"

    @model_validator(mode="after")
    def _unique_known_keys(self) -> Self:
        keys = self.requirement_keys
        if len(set(keys)) != len(keys):
            msg = "requirement_keys must be unique"
            raise ValueError(msg)
        labelled = [item.key for item in self.requirements or []]
        if len(set(labelled)) != len(labelled) or not set(labelled) <= set(keys):
            msg = "requirements need unique keys that are in requirement_keys"
            raise ValueError(msg)
        return self


class ExtractOfferEvidenceOutput(ContractPayload):
    """Output of ``extract_offer_evidence``: one entry per requested key.

    Keys without a text signal (platform, distance to attractions) should not
    be requested; the backend decides them. Keys whose meaning is not obvious
    need a label in ``requirements``.
    """

    evidence: list[RequirementEvidence] = Field(max_length=50)


# --- fetch_place_candidates --------------------------------------------------------


class FetchPlaceCandidatesInput(ContractPayload):
    """Input of ``fetch_place_candidates``: open data (OSM) for one city.

    Exactly one of ``city_query`` (free text) and ``city_slug`` is required.
    With ``city_query`` the worker derives ``city_slug`` by the rule in
    :data:`SLUG_PATTERN` and returns it in the output.
    """

    city_query: str | None = Field(default=None, min_length=1, max_length=200)
    city_slug: str | None = Field(default=None, pattern=SLUG_PATTERN, max_length=100)

    @model_validator(mode="after")
    def _exactly_one_city(self) -> Self:
        if (self.city_query is None) == (self.city_slug is None):
            msg = "exactly one of city_query and city_slug is required"
            raise ValueError(msg)
        return self


class FetchPlaceCandidatesOutput(ContractPayload):
    """Output of ``fetch_place_candidates`` (rows go to the places catalog).

    ``refreshed`` is ``False`` when the city already had candidates and nothing
    was fetched (``stored`` is then 0).
    """

    city_slug: str = Field(pattern=SLUG_PATTERN)
    source: Literal["osm"] = "osm"
    refreshed: bool
    stored: int = Field(ge=0)


# --- write_justifications ------------------------------------------------------------


class Justification(BaseModel):
    """A short reason a place is in the plan, like the ``explain()`` card.

    ``place_id`` is the catalog place id (the same id as in the plan item).
    ``profile_id`` is the person the card is written for; ``None`` is the
    group-level justification. ``source`` tells whether a model wrote ``text``
    or the deterministic template did.
    """

    model_config = ConfigDict(frozen=True)

    place_id: str = Field(min_length=1, max_length=100)
    profile_id: UUID | None = None
    text: str = Field(min_length=1, max_length=600)
    source: Literal["model", "template"]


class WriteJustificationsInput(ContractPayload):
    """Input of ``write_justifications``; the plan is read by id."""

    plan_id: UUID
    locale: Locale = "pl"
    provider: ProviderName = "openrouter"


class WriteJustificationsOutput(ContractPayload):
    """Output of ``write_justifications`` (also kept in ``job_results``)."""

    justifications: list[Justification] = Field(max_length=600)


# --- parse_expense_text and read_receipt (expenses) --------------------------------

CURRENCY_PATTERN: Final = r"^[A-Z]{3}$"


class ParseExpenseTextInput(ContractPayload):
    """Input of ``parse_expense_text``: one sentence typed by a trip member.

    The text is untrusted. The worker only extracts fields; the backend matches
    names to the trip's profiles and never saves a draft.
    """

    trip_id: UUID
    text: str = Field(min_length=1, max_length=500)
    locale: Locale = "pl"


class ParseExpenseTextOutput(ContractPayload):
    """Output of ``parse_expense_text`` (names exactly as written in the text).

    ``amount_minor`` is in minor units (grosze, cents). ``included_names`` lists
    people the text says took part; empty means everybody except
    ``excluded_names``.
    """

    amount_minor: int = Field(gt=0)
    currency: str | None = Field(default=None, pattern=CURRENCY_PATTERN)
    description: str = Field(default="", max_length=500)
    payer_name: str | None = Field(default=None, max_length=100)
    included_names: list[str] = Field(default_factory=list, max_length=50)
    excluded_names: list[str] = Field(default_factory=list, max_length=50)
    confidence: float | None = Field(default=None, ge=0, le=1)


ExpenseCategoryName = Literal[
    "food", "transport", "lodging", "activities", "shopping", "other"
]


class ReadReceiptInput(ContractPayload):
    """Input of ``read_receipt``: the image is read from ``expense_evidence``.

    The image never travels in the payload or the logs.
    """

    trip_id: UUID
    evidence_id: UUID


class ReadReceiptOutput(ContractPayload):
    """Output of ``read_receipt``: settlement fields only, never the image."""

    amount_minor: int = Field(gt=0)
    currency: str | None = Field(default=None, pattern=CURRENCY_PATTERN)
    spent_on: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    merchant: str | None = Field(default=None, max_length=200)
    category: ExpenseCategoryName | None = None
    needs_confirmation: bool
    reasons: list[str] = Field(default_factory=list, max_length=10)


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
    ``GET /api/v1/jobs/{id}`` (``status=ERROR``).
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


def document_not_found(document_id: UUID, kind: str) -> ContractError:
    """Error of a workflow whose pasted document is missing or of another kind.

    Args:
        document_id: Id from the payload.
        kind: Expected document kind (``plan`` or ``offer``).

    Returns:
        A ``ContractError`` with code ``document_not_found``.
    """
    data = ContractErrorData(
        code=ErrorCode.DOCUMENT_NOT_FOUND,
        supported_versions=sorted(SUPPORTED_CONTRACT_VERSIONS),
    )
    return ContractError(f"no {kind} document {document_id} for this trip", data)


def model_output_invalid(workflow: Workflow) -> ContractError:
    """Error of an LLM workflow whose model gave no valid structured output.

    Args:
        workflow: The workflow that asked the model.

    Returns:
        A ``ContractError`` with code ``model_output_invalid``.
    """
    data = ContractErrorData(
        code=ErrorCode.MODEL_OUTPUT_INVALID,
        supported_versions=sorted(SUPPORTED_CONTRACT_VERSIONS),
    )
    return ContractError(
        f"the model returned no valid answer for {workflow.value}", data
    )


def contract_failure(code: ErrorCode, message: str) -> ContractError:
    """Error of a workflow that cannot finish for a reason outside the payload shape.

    Args:
        code: Machine-readable reason.
        message: Text the backend shows in ``GET /jobs/{id}``.

    Returns:
        A ``ContractError`` with that code.
    """
    data = ContractErrorData(
        code=code, supported_versions=sorted(SUPPORTED_CONTRACT_VERSIONS)
    )
    return ContractError(message, data)


def not_implemented(workflow: Workflow) -> ContractError:
    """Error of a workflow that is in the contract but has no implementation yet.

    Args:
        workflow: The stub workflow.

    Returns:
        A ``ContractError`` with code ``not_implemented``.
    """
    data = ContractErrorData(
        code=ErrorCode.NOT_IMPLEMENTED,
        supported_versions=sorted(SUPPORTED_CONTRACT_VERSIONS),
    )
    return ContractError(f"workflow {workflow.value} is not implemented yet", data)


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
    # LLM workflows: enqueue on queue_for(provider), like generate_trip_plan.
    Workflow.PARSE_PASTED_PLAN: WorkflowSpec(
        Queue.OPENROUTER, ParsePastedPlanInput, ParsePastedPlanOutput
    ),
    Workflow.EXTRACT_OFFER_EVIDENCE: WorkflowSpec(
        Queue.OPENROUTER, ExtractOfferEvidenceInput, ExtractOfferEvidenceOutput
    ),
    Workflow.WRITE_JUSTIFICATIONS: WorkflowSpec(
        Queue.OPENROUTER, WriteJustificationsInput, WriteJustificationsOutput
    ),
    # Expenses: the local model on the GB10 reads the text or the image.
    Workflow.PARSE_EXPENSE_TEXT: WorkflowSpec(
        Queue.LOCAL_LLM, ParseExpenseTextInput, ParseExpenseTextOutput
    ),
    Workflow.READ_RECEIPT: WorkflowSpec(
        Queue.LOCAL_LLM, ReadReceiptInput, ReadReceiptOutput
    ),
    # Open data (OSM), no LLM.
    Workflow.FETCH_PLACE_CANDIDATES: WorkflowSpec(
        Queue.DEFAULT, FetchPlaceCandidatesInput, FetchPlaceCandidatesOutput
    ),
}

EVENTS: Final[Mapping[str, type[BaseModel]]] = {PROGRESS_EVENT: Progress}

SCHEDULED_WORKFLOWS: Final[Mapping[str, str]] = {
    "heartbeat": "*/30 * * * * *",
    "reset_demo_account": "0 0 4 * * *",
    "purge_notifications": "0 30 3 * * *",
}
"""Internal scheduled workflows (name -> 6-field cron, evaluated in
``SCHEDULE_TIMEZONE``), never enqueued by the backend. ``heartbeat`` upserts
``worker_heartbeats`` every 30 seconds; ``reset_demo_account`` restores the
jury's demo account at 04:00; ``purge_notifications`` deletes old rows of
``notifications`` at 03:30."""

SCHEDULE_TIMEZONE: Final = "Europe/Warsaw"
"""IANA timezone of every cron expression above."""


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
