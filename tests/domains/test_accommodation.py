"""Accommodation domain: offer quotes and their assessment, no real model."""

from typing import Any
from uuid import UUID, uuid4

import pytest
from dbos import DBOS, DBOSClient, PortableWorkflowError
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy.dialects import postgresql

from tests.helpers import enqueue
from tuttitrip_worker.accommodation import steps
from tuttitrip_worker.accommodation.logic.evidence import (
    build_evidence,
    verified_quotes,
)
from tuttitrip_worker.accommodation.schemas import ExtractedQuotes, RequirementQuotes
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    ErrorCode,
    ExtractOfferEvidenceOutput,
    Workflow,
)
from tuttitrip_worker.quotes import find_quote
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.llm.models import catalog

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


# --- workflow -----------------------------------------------------------------


class FakeModels:
    """One FunctionModel that plays the extractor and the judge."""

    def __init__(
        self,
        quotes: dict[str, list[str]],
        verdicts: dict[str, tuple[str, float | None]] | None = None,
        *,
        judge_down: bool = False,
    ) -> None:
        self.quotes = quotes
        self.verdicts = verdicts or {}
        self.judge_down = judge_down
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
        if self.judge_down:
            raise ModelAPIError(model_name="basal", message="down")
        quote = prompt.split("Quote from the offer: ", 1)[1]
        verdict, confidence = self.verdicts[quote]
        details = {} if confidence is None else {"confidence": {"verdict": confidence}}
        return ModelResponse(
            parts=[ToolCallPart(tool.name, {"verdict": verdict})],
            provider_details=details or None,
        )


@pytest.fixture
def offer_text(monkeypatch: pytest.MonkeyPatch) -> dict[str, str | None]:
    holder: dict[str, str | None] = {"text": OFFER}

    async def fake_load(document_id: UUID, trip_id: UUID) -> str | None:
        del document_id, trip_id
        return holder["text"]

    monkeypatch.setattr(steps, "load_offer_text", fake_load)
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
    client: DBOSClient, dbos: Settings, models: FakeModels, body: dict[str, Any]
) -> tuple[ExtractOfferEvidenceOutput, str]:
    with catalog.override(FunctionModel(models)):
        handle = enqueue(client, dbos, Workflow.EXTRACT_OFFER_EVIDENCE, body)
        output = ExtractOfferEvidenceOutput.model_validate(handle.get_result())
    return output, handle.get_workflow_id()


def test_silent_offer_and_absent_quote_and_seasonal_quote(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, str | None],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del offer_text
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
    assert [(q.text, q.verdict, q.confidence) for q in by_key["pool"]] == [
        ("Brak basenu na terenie obiektu.", "absent", 0.9)
    ]
    assert by_key["spa"][0].verdict == "not_applicable"
    assert by_key["spa"][0].confidence == pytest.approx(0.2)
    assert by_key["gym"] == []  # a quote from outside the text is dropped
    assert by_key["bar"] == []  # no quote, no assessment
    assert all(q.text in OFFER for qs in by_key.values() for q in qs)
    assert "pool (Basen)" in models.prompts[0]  # label reaches the judge
    assert len(models.prompts) == 2  # nothing is judged without a quote
    assert saved == [
        (workflow_id, "extract_offer_evidence", output.model_dump(mode="json"))
    ]
    steps_run = [s["function_name"] for s in DBOS.list_workflow_steps(workflow_id)]
    assert "offer_quote_extractor__model.request" in steps_run
    assert "offer_requirement_judge__model.request" in steps_run


def test_offer_without_the_word_pool_gives_no_quote_and_no_verdict(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, str | None],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del saved
    offer_text["text"] = "Apartament z balkonem i widokiem na rzekę."
    output, _ = run(client, dbos, FakeModels({"pool": []}), payload("pool"))
    assert output.evidence[0].quotes == []


def test_without_a_decision_model_quotes_come_back_unassessed(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, str | None],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del offer_text, saved
    models = FakeModels({"pool": ["Brak basenu na terenie obiektu."]}, judge_down=True)
    output, _ = run(client, dbos, models, payload("pool"))
    quote = output.evidence[0].quotes[0]
    assert quote.text == "Brak basenu na terenie obiektu."
    assert (quote.verdict, quote.confidence) == (None, None)


def test_fallback_language_model_may_report_no_confidence(
    client: DBOSClient,
    dbos: Settings,
    offer_text: dict[str, str | None],
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
    offer_text: dict[str, str | None],
    saved: list[tuple[str, str, dict[str, Any]]],
) -> None:
    del saved
    offer_text["text"] = None
    with catalog.override(FunctionModel(FakeModels({}))):
        handle = enqueue(client, dbos, Workflow.EXTRACT_OFFER_EVIDENCE, payload("pool"))
        with pytest.raises(PortableWorkflowError) as info:
            handle.get_result()
    assert info.value.code == ErrorCode.DOCUMENT_NOT_FOUND.value
