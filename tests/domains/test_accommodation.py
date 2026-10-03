"""Accommodation domain: offer quotes and their assessment, no real model."""

import logging
import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, get_args
from uuid import UUID, uuid4

import pytest
from dbos import DBOS, DBOSClient, PortableWorkflowError
from pydantic import SecretStr
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.system_one import SystemOneModel
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from tests.helpers import enqueue
from tuttitrip_worker.accommodation import agents, steps
from tuttitrip_worker.accommodation.logic.evidence import (
    MAX_QUOTES_PER_KEY,
    build_evidence,
    verified_quotes,
)
from tuttitrip_worker.accommodation.schemas import ExtractedQuotes, RequirementQuotes
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    ErrorCode,
    ExtractOfferEvidenceOutput,
    OfferVerdict,
    Workflow,
)
from tuttitrip_worker.quotes import find_quote
from tuttitrip_worker.shared.config.settings import LlmSettings, Settings
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.llm.models import ModelKey, build_model, catalog, model_id

OFFER = (
    "Apartament Stare Miasto, 2 sypialnie.\n"
    "Brak basenu na terenie obiektu.\n"
    "Bezpłatny parking przy budynku.\n"
    "Sauna czynna tylko latem.\n"
)


def extracted(**quotes: list[str]) -> ExtractedQuotes:
    return ExtractedQuotes(
        requirements=[
            RequirementQuotes(requirement_key=key, quotes=found)
            for key, found in quotes.items()
        ]
    )


# --- pure logic ---------------------------------------------------------------


def test_find_quote_is_exact_but_tolerates_whitespace_runs() -> None:
    assert find_quote(OFFER, "Brak basenu na terenie obiektu.") is not None
    assert find_quote(OFFER, "Brak  basenu\nna terenie obiektu.") == (
        "Brak basenu na terenie obiektu."
    )
    assert find_quote("Brak  basenu\nna terenie", "Brak basenu na terenie") == (
        "Brak  basenu\nna terenie"
    )
    assert find_quote(OFFER, "brak basenu") is None  # case counts
    assert find_quote(OFFER, "Jest basen") is None
    assert find_quote(OFFER, "   ") is None


def test_find_quote_accepts_non_breaking_spaces() -> None:
    assert find_quote("Brak\xa0basenu\u202fna terenie", "Brak basenu na terenie") == (
        "Brak\xa0basenu\u202fna terenie"
    )
    assert find_quote("Brak basenu", "Brak\xa0basenu") == "Brak basenu"


def test_find_quote_does_not_cut_words_and_rejects_tiny_quotes() -> None:
    assert find_quote("Whirlpool na dachu", "pool") is None
    assert find_quote("Whirlpool na dachu", "Whirlpool") == "Whirlpool"
    assert find_quote("Pool na dachu", "pool") is None  # case
    assert find_quote("Pool: tak", "Pool") == "Pool"
    assert find_quote("Bar 24/7", "24/7") == "24/7"  # symbol edge needs no boundary
    assert find_quote("a b c", "a") is None
    assert find_quote("ab c", "ab") is None
    assert find_quote("abc", "abc") == "abc"


def test_find_quote_returns_the_first_occurrence() -> None:
    assert find_quote("Sauna, tak. Sauna, nie.", "Sauna") == "Sauna"
    assert find_quote("x Pool, tak. Pool, nie.", "Pool, nie.") == "Pool, nie."


def test_quotes_outside_the_offer_are_dropped() -> None:
    result = verified_quotes(
        OFFER,
        extracted(
            pool=["Brak basenu na terenie obiektu.", "Basen z widokiem na góry"],
            parking=[
                "Bezpłatny parking przy budynku.",
                "Bezpłatny parking przy budynku.",
            ],
            spa=["Sauna czynna tylko latem."],
        ),
        ["pool", "parking"],
    )
    assert result == {
        "pool": ["Brak basenu na terenie obiektu."],
        "parking": ["Bezpłatny parking przy budynku."],  # duplicate removed
    }  # "spa" was not requested


def test_verified_quotes_keep_at_most_three_and_no_overlong_ones() -> None:
    offer = "a1. a2. a3. a4. " + "x" * 1001
    found = verified_quotes(
        offer, extracted(k=["a1.", "a2.", "a3.", "a4.", "x" * 1001]), ["k"]
    )
    assert found == {"k": ["a1.", "a2.", "a3."]}
    assert verified_quotes(offer, extracted(k=["x" * 1001]), ["k"]) == {"k": []}


def test_build_evidence_has_every_key_and_assessments_are_optional() -> None:
    evidence = build_evidence(
        ["pool", "parking"],
        {"pool": ["Brak basenu"], "parking": []},
        {("pool", "Brak basenu"): ("absent", 0.8)},
    )
    assert [e.requirement_key for e in evidence] == ["pool", "parking"]
    assert evidence[0].quotes[0].verdict == "absent"
    assert evidence[0].quotes[0].confidence == pytest.approx(0.8)
    assert evidence[1].quotes == []
    bare = build_evidence(["pool"], {"pool": ["Brak basenu"]}, {})
    assert bare[0].quotes[0].verdict is None
    assert bare[0].quotes[0].confidence is None


def test_offer_query_selects_only_an_offer_of_the_trip() -> None:
    sql = str(steps.offer_query(uuid4(), uuid4()).compile(dialect=postgresql.dialect()))
    assert "pasted_documents.text" in sql
    assert "pasted_documents.id =" in sql
    assert "pasted_documents.trip_id =" in sql
    assert "pasted_documents.kind =" in sql


# --- agents and contract ------------------------------------------------------


def test_verdict_values_are_the_contracts_offer_verdicts() -> None:
    assert {v.value for v in agents.Verdict} == set(get_args(OfferVerdict))


def test_quote_slots_match_the_quote_limit() -> None:
    assert set(agents.QuoteAssessments.model_fields) == set(agents.QUOTE_SLOTS)
    assert len(agents.QUOTE_SLOTS) == MAX_QUOTES_PER_KEY


def test_extractor_uses_chat_and_judge_uses_decide_with_chat_fallback() -> None:
    assert agents.quote_extractor.model == model_id(ModelKey.CHAT)
    assert agents.requirement_judge.model == model_id(ModelKey.DECIDE)
    settings = LlmSettings(
        gb10_api_key=SecretStr("k"), openrouter_api_key=SecretStr("o")
    )
    chain = build_model(ModelKey.DECIDE, settings)
    assert isinstance(chain, FallbackModel)
    first, last = chain.models[0], chain.models[-1]
    assert isinstance(first, SystemOneModel)
    assert isinstance(last, OpenAIChatModel)
    assert last.model_name == settings.gb10_chat_model
    chat = build_model(ModelKey.CHAT, settings)
    assert isinstance(chat, FallbackModel)
    assert chat.models[0].model_name == settings.gb10_chat_model


# --- workflow -----------------------------------------------------------------


class FakeModels:
    """One FunctionModel that plays the extractor and the judge."""

    def __init__(
        self,
        quotes: dict[str, list[str]],
        verdicts: dict[str, tuple[str, float | None]] | None = None,
        *,
        judge_error: Exception | None = None,
    ) -> None:
        self.quotes = quotes
        self.verdicts = verdicts or {}
        self.judge_error = judge_error
        self.prompts: list[str] = []

    def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        prompt = str(messages[-1].parts[-1].content)  # type: ignore[union-attr]  # ty: ignore[unresolved-attribute]
        if "requirements" in tool.parameters_json_schema["properties"]:
            args: dict[str, Any] = {
                "requirements": [
                    {"requirement_key": key, "quotes": found}
                    for key, found in self.quotes.items()
                ]
            }
            return ModelResponse(parts=[ToolCallPart(tool.name, args)])
        self.prompts.append(prompt)
        if self.judge_error is not None:
            raise self.judge_error
        answers: dict[str, str] = {}
        confidence: dict[str, float] = {}
        for number in (1, 2, 3):
            tag = f"quote_{number}"
            found = re.search(rf"<{tag}>(.*?)</{tag}>", prompt, re.DOTALL)
            text = found.group(1) if found else ""
            verdict, level = self.verdicts.get(text, ("not_applicable", None))
            answers[tag] = verdict
            if level is not None:
                confidence[tag] = level
        details = {"confidence": confidence} if confidence else None
        return ModelResponse(
            parts=[ToolCallPart(tool.name, answers)], provider_details=details
        )


class FakeConnection:
    """Stands in for the Postgres connection of the real ``load_offer_text``."""

    def __init__(self, text: str | None, seen: list[Any]) -> None:
        self.text = text
        self.seen = seen

    async def execute(self, query: Select[Any]) -> SimpleNamespace:
        self.seen.append(query.compile().params)
        return SimpleNamespace(scalar_one_or_none=lambda: self.text)


@pytest.fixture
def offer_text(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run the REAL ``load_offer_text`` step against a fake transaction."""
    holder: dict[str, Any] = {"text": OFFER, "seen": []}

    @asynccontextmanager
    async def fake_transaction() -> AsyncGenerator[FakeConnection]:
        yield FakeConnection(holder["text"], holder["seen"])

    monkeypatch.setattr(steps, "transaction", fake_transaction)
    return holder


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, Any]]]:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_save(workflow_id: str, name: str, result: dict[str, Any]) -> None:
        calls.append((workflow_id, name, result))

    monkeypatch.setattr(job_results, "save_job_result", fake_save)
    return calls


def payload(*keys: str, **extra: object) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "trip_id": str(uuid4()),
        "document_id": str(uuid4()),
        "requirement_keys": list(keys),
        **extra,
    }


def run(
    client: DBOSClient,
    dbos: Settings,
    models: FakeModels,
    body: dict[str, Any],
    override: Model | None = None,
) -> tuple[ExtractOfferEvidenceOutput, str]:
    with catalog.override(override or FunctionModel(models)):
        handle = enqueue(client, dbos, Workflow.EXTRACT_OFFER_EVIDENCE, body)
        output = ExtractOfferEvidenceOutput.model_validate(handle.get_result())
    return output, handle.get_workflow_id()


def test_silent_offer_and_absent_quote_and_seasonal_quote(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    models = FakeModels(
        {
            "pool": ["Brak basenu na terenie obiektu."],
            "spa": ["Sauna czynna tylko latem."],
            "gym": ["Siłownia 24/7"],  # invented: not in the offer
            "bar": [],  # offer is silent
        },
        {
            "Brak basenu na terenie obiektu.": ("absent", 0.9),
            "Sauna czynna tylko latem.": ("not_applicable", 0.2),
        },
    )
    body = payload(
        "pool",
        "spa",
        "gym",
        "bar",
        requirements=[{"key": "pool", "label": "Basen"}],
    )
    output, workflow_id = run(client, dbos, models, body)
    by_key = {e.requirement_key: e.quotes for e in output.evidence}

    assert [e.requirement_key for e in output.evidence] == ["pool", "spa", "gym", "bar"]
    assert [(q.text, q.verdict) for q in by_key["pool"]] == [
        ("Brak basenu na terenie obiektu.", "absent")
    ]
    assert by_key["pool"][0].confidence == pytest.approx(0.9)
    assert by_key["spa"][0].verdict == "not_applicable"
    assert by_key["spa"][0].confidence == pytest.approx(0.2)
    assert by_key["gym"] == []  # a quote from outside the text is dropped
    assert by_key["bar"] == []  # no quote, no assessment
    assert all(q.text in OFFER for qs in by_key.values() for q in qs)
    assert "pool (Basen)" in models.prompts[0]  # label reaches the judge
    assert "<quote_1>Brak basenu" in models.prompts[0]  # quote is wrapped as data
    assert len(models.prompts) == 2  # one call per key with quotes
    assert saved == [
        (workflow_id, "extract_offer_evidence", output.model_dump(mode="json"))
    ]
    steps_run = [s["function_name"] for s in DBOS.list_workflow_steps(workflow_id)]
    assert "load_offer_text" in steps_run  # the real step ran
    assert "offer_quote_extractor__model.request" in steps_run
    assert "offer_requirement_judge__model.request" in steps_run
    (params,) = offer_text["seen"]  # ids reached SQL as UUIDs
    assert {type(v) for v in params.values() if not isinstance(v, str)} <= {UUID}
    assert UUID(body["document_id"]) in params.values()


def test_all_quotes_of_a_key_are_judged_in_one_call(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del offer_text, saved
    models = FakeModels(
        {"pool": ["Brak basenu na terenie obiektu.", "Sauna czynna tylko latem."]},
        {
            "Brak basenu na terenie obiektu.": ("absent", 0.9),
            "Sauna czynna tylko latem.": ("present", 0.6),
        },
    )
    output, _ = run(client, dbos, models, payload("pool"))
    assert len(models.prompts) == 1
    # The worker does not aggregate: conflicting verdicts are both returned.
    assert [q.verdict for q in output.evidence[0].quotes] == ["absent", "present"]


def test_offer_without_the_word_pool_gives_no_quote_and_no_verdict(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del saved
    offer_text["text"] = "Apartament z balkonem i widokiem na rzekę."
    output, _ = run(client, dbos, FakeModels({"pool": []}), payload("pool"))
    assert output.evidence[0].quotes == []


def test_provider_failure_keeps_quotes_unassessed_and_logs_a_warning(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    del offer_text, saved
    down = ModelAPIError(model_name="basal", message="down")
    models = FakeModels({"pool": ["Brak basenu na terenie obiektu."]}, judge_error=down)
    with caplog.at_level(logging.WARNING):
        output, _ = run(client, dbos, models, payload("pool"))
    quote = output.evidence[0].quotes[0]
    assert quote.text == "Brak basenu na terenie obiektu."
    assert (quote.verdict, quote.confidence) == (None, None)
    assert "offer judge unavailable for pool" in caplog.text


def test_every_fallback_model_down_is_judge_unavailable(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del offer_text, saved
    down = ModelAPIError(model_name="m", message="down")
    models = FakeModels({"pool": ["Brak basenu na terenie obiektu."]}, judge_error=down)
    chain = FallbackModel(FunctionModel(models), FunctionModel(models))
    output, _ = run(client, dbos, models, payload("pool"), override=chain)
    assert output.evidence[0].quotes[0].verdict is None


def test_misconfiguration_fails_the_job(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del offer_text, saved
    models = FakeModels(
        {"pool": ["Brak basenu na terenie obiektu."]},
        judge_error=UserError("No API key is configured"),
    )
    with catalog.override(FunctionModel(models)):
        handle = enqueue(client, dbos, Workflow.EXTRACT_OFFER_EVIDENCE, payload("pool"))
        with pytest.raises(PortableWorkflowError):
            handle.get_result()


def test_fallback_language_model_may_report_no_confidence(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del offer_text, saved
    models = FakeModels(
        {"parking": ["Bezpłatny parking przy budynku."]},
        {"Bezpłatny parking przy budynku.": ("present", None)},
    )
    output, _ = run(client, dbos, models, payload("parking"))
    quote = output.evidence[0].quotes[0]
    assert (quote.verdict, quote.confidence) == ("present", None)


def test_missing_offer_ends_with_document_not_found(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, Any],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del saved
    offer_text["text"] = None
    with catalog.override(FunctionModel(FakeModels({}))):
        handle = enqueue(client, dbos, Workflow.EXTRACT_OFFER_EVIDENCE, payload("pool"))
        with pytest.raises(PortableWorkflowError) as info:
            handle.get_result()
    assert info.value.code == ErrorCode.DOCUMENT_NOT_FOUND.value
