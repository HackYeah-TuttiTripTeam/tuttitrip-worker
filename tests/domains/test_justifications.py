"""write_justifications: facts from a stored plan, the checks, FunctionModel only."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import pytest
from dbos import DBOS, DBOSClient, PortableWorkflowError
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from sqlalchemy import Select
from sqlalchemy.dialects import postgresql

from tests.helpers import enqueue
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    ErrorCode,
    Workflow,
    WriteJustificationsOutput,
)
from tuttitrip_worker.planning import steps
from tuttitrip_worker.planning.logic.justification_check import (
    check_batch,
    check_text,
    passing,
)
from tuttitrip_worker.planning.logic.verdict_facts import build_facts, split_batches
from tuttitrip_worker.planning.schemas import JustificationItem
from tuttitrip_worker.shared.config.settings import Settings, get_settings
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.llm.models import catalog

KASIA, TOMEK, OLA = str(uuid4()), str(uuid4()), str(uuid4())
WAWEL, KOPALNIA, RYNEK = "place-wawel", "place-kopalnia", "place-rynek"


def person(profile_id: str, name: str, u: float) -> dict[str, Any]:
    return {"profile_id": profile_id, "name": name, "u": u}


def stop(place_id: str, name: str) -> dict[str, Any]:
    return {"place_id": place_id, "name": name}


RESULT: dict[str, Any] = {
    "fairness": {
        "per_person": [
            person(KASIA, "Kasia", 61.0),
            person(TOMEK, "Tomek", 70.0),
            person(OLA, "Ola", 55.0),
        ]
    },
    "days": [{"items": [stop(WAWEL, "Zamek na Wawelu"), stop(RYNEK, "Rynek 2")]}],
    "explain": [
        {
            "place_id": KOPALNIA,
            "profile_id": TOMEK,
            "match": 0.82,
            "effort": 0.4,
            "utility": 74.5,
        },
        {
            "place_id": KOPALNIA,
            "profile_id": KASIA,
            "match": 0.1,
            "effort": 0.9,
            "utility": 3.0,
        },
    ],
    "verdicts": [
        {
            "place_id": WAWEL,
            "verdict": "fits",
            "v_p": 0.7,
            "yes": [{"profile_id": TOMEK}, {"profile_id": OLA}],
            "no": [],
            "skip_codes": [],
            "substitute_place_id": None,
        },
        {
            "place_id": KOPALNIA,
            "verdict": "skip",
            "v_p": -0.3,
            "yes": [{"profile_id": TOMEK}],
            "no": [{"profile_id": KASIA, "reason_code": "other"}],
            "skip_codes": ["veto"],
            "substitute_place_id": RYNEK,
        },
    ],
}

GOOD_SKIP = (
    "Pomijamy to miejsce, bo Kasi nie pasuje i zgłosiła weto, a Tomek był na tak."
)
GOOD_FITS = "Zamek na Wawelu pasuje: Tomek i Ola są na tak (0,7)."


def text_item(place_id: str, text: str) -> dict[str, str]:
    return {"place_id": place_id, "text": text}


# --- facts from the stored plan -----------------------------------------------


def test_facts_use_names_and_keep_the_numbers_of_the_algorithm() -> None:
    wawel, kopalnia = build_facts(RESULT)

    assert wawel.place_name == "Zamek na Wawelu"
    assert [p.name for p in wawel.yes] == ["Tomek", "Ola"]
    assert kopalnia.place_name is None  # a skipped place is not in the days
    assert kopalnia.substitute_name == "Rynek 2"
    assert [(p.name, p.reason_code) for p in kopalnia.no] == [("Kasia", "other")]
    assert kopalnia.skip_codes == ["veto"]
    assert [(p.name, p.match, p.utility) for p in kopalnia.people] == [
        ("Tomek", 0.82, 74.5),
        ("Kasia", 0.1, 3.0),
    ]


def test_a_plan_without_verdicts_has_no_facts() -> None:
    assert build_facts({"verdicts": None}) == []
    assert build_facts({}) == []


def test_people_outside_the_ledger_are_dropped() -> None:
    result = {**RESULT, "fairness": {"per_person": [person(KASIA, "Kasia", 61.0)]}}
    assert [p.name for p in build_facts(result)[0].yes] == []


def test_batches_are_consecutive_and_never_empty() -> None:
    facts = build_facts(RESULT)
    assert [len(b) for b in split_batches(facts, 1)] == [1, 1]
    assert [len(b) for b in split_batches(facts, 5)] == [2]
    assert split_batches([], 3) == []


# --- the checks ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        GOOD_SKIP,
        "Weto Kasi wyklucza to miejsce.",
        "Tomek ma dopasowanie 0,82 i wysiłek 0.4.",
        "Dopasowanie Tomka to 82%, a Kasi 10%.",
        "Użyteczność Tomka wynosi 74,5, a Kasi 3.",
        "Za tym miejscem jest 1 osoba, przeciw 1 osoba.",
        "Zamiast tego Rynek 2 jest bliżej.",
    ],
)
def test_texts_made_of_the_given_data_pass(text: str) -> None:
    kopalnia = build_facts(RESULT)[1]
    assert check_text(text, kopalnia) == []


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("Bilet kosztuje 45 zł.", "the number 45"),
        ("Dopasowanie Tomka to 0,83.", "the number 0,83"),
        ("Do centrum jest 12 km.", "the number 12"),
        ("Pomijamy, bo Marek ma weto.", "the name Marek"),
        ("Pomijamy to, bo Kasia nie chce. Tomek tak. A Ola nie wie.", "more than 2"),
        ("  ", "empty"),
        ("Bo Kasia. " + "x" * 400, "longer than"),
    ],
)
def test_texts_with_a_number_or_name_outside_the_data_fail(
    text: str, problem: str
) -> None:
    kopalnia = build_facts(RESULT)[1]
    assert any(problem in found for found in check_text(text, kopalnia))


def test_the_first_word_of_a_sentence_is_not_taken_for_a_name() -> None:
    kopalnia = build_facts(RESULT)[1]
    assert check_text("Pomijamy to. Weto zgłosiła Kasia.", kopalnia) == []


def test_a_batch_needs_one_entry_per_place_and_no_other() -> None:
    facts = build_facts(RESULT)
    items = [
        JustificationItem(place_id=WAWEL, text=GOOD_FITS),
        JustificationItem(place_id=WAWEL, text=GOOD_FITS),
        JustificationItem(place_id="other", text="Ok."),
    ]
    problems = check_batch(items, facts)
    assert problems == [
        "other: not a place of this request",
        f"{KOPALNIA}: missing",
        f"{WAWEL}: written twice",
    ]
    assert [i.place_id for i in passing(items, facts)] == [WAWEL]


# --- the step -----------------------------------------------------------------


def test_plan_select_reads_the_result_of_one_version() -> None:
    sql = str(steps.build_plan_select(uuid4()).compile(dialect=postgresql.dialect()))
    assert "plan_versions.result" in sql
    assert "plan_versions.id =" in sql


# --- the workflow -------------------------------------------------------------


class FakeResult:
    def __init__(self, value: dict[str, Any] | None) -> None:
        self.value = value

    def scalar(self) -> dict[str, Any] | None:
        return self.value


class FakeConnection:
    def __init__(self, env: Env) -> None:
        self.env = env

    async def execute(self, statement: Select[Any]) -> FakeResult:
        self.env.selects.append(str(statement.compile(dialect=postgresql.dialect())))
        return FakeResult(self.env.result)


class Env:
    """Fake database plus the recorded model calls."""

    def __init__(self) -> None:
        self.result: dict[str, Any] | None = RESULT
        self.selects: list[str] = []
        self.saved: list[tuple[str, str, dict[str, Any]]] = []
        self.prompts: list[str] = []
        self.retries: list[str] = []
        self.answers: list[list[dict[str, str]]] = [
            [text_item(WAWEL, GOOD_FITS), text_item(KOPALNIA, GOOD_SKIP)]
        ]

    def model(self) -> FunctionModel:
        def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.prompts.append(str(getattr(messages[0].parts[-1], "content", "")))
            last = str(messages[-1].parts[-1])
            if "Fix these justifications" in last:
                self.retries.append(last)
            index = min(len(self.prompts), len(self.answers)) - 1
            tool = info.output_tools[0].name
            return ModelResponse(
                parts=[ToolCallPart(tool, {"items": self.answers[index]})]
            )

        return FunctionModel(answer)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> Env:
    """Replace the engine under the real ``load_verdict_facts`` step."""
    state = Env()

    @asynccontextmanager
    async def fake_transaction() -> AsyncGenerator[FakeConnection]:
        yield FakeConnection(state)

    async def fake_save(workflow_id: str, name: str, result: dict[str, Any]) -> None:
        state.saved.append((workflow_id, name, result))

    monkeypatch.setattr(steps, "transaction", fake_transaction)
    monkeypatch.setattr(job_results, "save_job_result", fake_save)
    return state


def payload(locale: str = "pl") -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "plan_id": str(uuid4()),
        "locale": locale,
    }


def run(
    client: DBOSClient,
    dbos: Settings,
    env: Env,
    body: dict[str, Any],
    *,
    workflow_id: str | None = None,
) -> tuple[str, WriteJustificationsOutput]:
    with catalog.override(env.model()):
        handle = enqueue(
            client, dbos, Workflow.WRITE_JUSTIFICATIONS, body, workflow_id=workflow_id
        )
        output = WriteJustificationsOutput.model_validate(handle.get_result())
    return handle.get_workflow_id(), output


def test_a_skip_with_a_veto_is_worded_from_the_data(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    workflow_id, output = run(client, dbos, env, payload())

    by_place = {j.place_id: j for j in output.justifications}
    assert by_place[KOPALNIA].text == GOOD_SKIP
    assert "weto" in by_place[KOPALNIA].text
    assert {j.source for j in output.justifications} == {"model"}
    assert {j.profile_id for j in output.justifications} == {None}
    assert len(env.selects) == 1
    assert "plan_versions" in env.selects[0]
    assert "Kasia" in env.prompts[0]  # the model saw the names, not profile ids
    assert KASIA not in env.prompts[0]
    assert "Polish" in env.prompts[0]
    assert env.saved == [
        (workflow_id, "write_justifications", output.model_dump(mode="json"))
    ]
    assert client.get_event(workflow_id, "progress") == {
        "stage": "done",
        "percent": 100,
    }
    steps_run = [s["function_name"] for s in DBOS.list_workflow_steps(workflow_id)]
    assert "verdict_justifier__model.request" in steps_run


def test_the_language_follows_the_locale(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    run(client, dbos, env, payload("en"))
    assert "English" in env.prompts[0]


def test_a_number_outside_the_data_makes_the_agent_retry(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    invented = "Pomijamy, bo bilet kosztuje 45 zł, a Kasia ma weto."
    env.answers = [
        [text_item(WAWEL, GOOD_FITS), text_item(KOPALNIA, invented)],
        [text_item(WAWEL, GOOD_FITS), text_item(KOPALNIA, GOOD_SKIP)],
    ]
    _, output = run(client, dbos, env, payload())

    assert len(env.prompts) == 2
    assert "the number 45" in env.retries[0]
    assert {j.text for j in output.justifications} == {GOOD_FITS, GOOD_SKIP}


def test_retries_run_out_and_the_bad_text_is_left_to_the_template(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    env.answers = [
        [
            text_item(WAWEL, GOOD_FITS),
            text_item(KOPALNIA, "Pomijamy, bo dojazd to 45 minut."),
        ]
    ]
    _, output = run(client, dbos, env, payload())

    assert len(env.prompts) == 3  # first try plus two retries
    assert [j.place_id for j in output.justifications] == [WAWEL]


def test_verdicts_are_sent_in_batches(
    client: DBOSClient,
    dbos: Settings,
    env: Env,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TUTTITRIP_PLANNING__JUSTIFICATION_BATCH_SIZE", "1")

    get_settings.cache_clear()
    env.answers = [
        [text_item(WAWEL, GOOD_FITS)],
        [text_item(KOPALNIA, GOOD_SKIP)],
    ]
    _, output = run(client, dbos, env, payload())

    assert len(env.prompts) == 2
    assert WAWEL in env.prompts[0]
    assert KOPALNIA in env.prompts[1]
    assert len(output.justifications) == 2


def test_the_same_workflow_id_calls_the_model_once(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    body = payload()
    key = f"write_justifications:{body['plan_id']}"
    first_id, first = run(client, dbos, env, body, workflow_id=key)
    second_id, second = run(client, dbos, env, body, workflow_id=key)

    assert first_id == second_id == key
    assert first == second
    assert len(env.prompts) == 1
    assert len(env.saved) == 1


def test_a_plan_without_verdicts_gives_an_empty_result(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    env.result = {**RESULT, "verdicts": None}
    _, output = run(client, dbos, env, payload())
    assert output.justifications == []
    assert env.prompts == []


def test_an_unknown_plan_ends_with_document_not_found(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    env.result = None
    with catalog.override(env.model()):
        handle = enqueue(client, dbos, Workflow.WRITE_JUSTIFICATIONS, payload())
        with pytest.raises(PortableWorkflowError) as info:
            handle.get_result()
    assert info.value.code == ErrorCode.DOCUMENT_NOT_FOUND.value
    assert env.prompts == []


def test_a_payload_without_a_plan_is_invalid(
    client: DBOSClient, dbos: Settings, env: Env
) -> None:
    handle = enqueue(
        client, dbos, Workflow.WRITE_JUSTIFICATIONS, {"contract_version": 1}
    )
    with pytest.raises(PortableWorkflowError) as info:
        handle.get_result()
    assert info.value.code == ErrorCode.INVALID_PAYLOAD.value
    assert env.prompts == []
