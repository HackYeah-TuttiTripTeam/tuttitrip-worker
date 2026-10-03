"""Contract stubs: workflows in the contract whose implementation comes later."""

from typing import Any
from uuid import uuid4

import pytest
from dbos import DBOSClient, PortableWorkflowError

from tests.helpers import enqueue
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    WORKFLOWS,
    ErrorCode,
    ExtractOfferEvidenceInput,
    FetchPlaceCandidatesInput,
    ParsePastedPlanInput,
    ParsePastedPlanOutput,
    Queue,
    Workflow,
    WriteJustificationsInput,
    queue_for,
)
from tuttitrip_worker.shared.config.settings import Settings

STUBS: dict[Workflow, dict[str, Any]] = {
    Workflow.PARSE_PASTED_PLAN: {
        "trip_id": str(uuid4()),
        "document_id": str(uuid4()),
        "city_slug": "krakow",
    },
    Workflow.EXTRACT_OFFER_EVIDENCE: {
        "trip_id": str(uuid4()),
        "document_id": str(uuid4()),
        "requirement_keys": ["pool", "parking"],
    },
    Workflow.FETCH_PLACE_CANDIDATES: {"city_slug": "krakow"},
    Workflow.WRITE_JUSTIFICATIONS: {"plan_id": str(uuid4())},
}


@pytest.mark.parametrize("name", list(STUBS))
def test_stub_ends_with_not_implemented(
    client: DBOSClient, dbos: Settings, name: Workflow
) -> None:
    payload = {"contract_version": CONTRACT_VERSION, **STUBS[name]}
    handle = enqueue(client, dbos, name, payload)
    with pytest.raises(PortableWorkflowError) as info:
        handle.get_result()
    assert info.value.name == "ContractError"
    assert info.value.code == ErrorCode.NOT_IMPLEMENTED.value


def test_stub_still_validates_the_payload(client: DBOSClient, dbos: Settings) -> None:
    payload = {"contract_version": CONTRACT_VERSION, "city_slug": "krakow"}
    handle = enqueue(client, dbos, Workflow.PARSE_PASTED_PLAN, payload)
    with pytest.raises(PortableWorkflowError) as info:
        handle.get_result()
    assert info.value.code == ErrorCode.INVALID_PAYLOAD.value


def test_new_workflows_are_a_compatible_change() -> None:
    assert CONTRACT_VERSION == 1
    assert {1} == SUPPORTED_CONTRACT_VERSIONS


def test_llm_workflows_use_the_provider_queue_and_osm_the_default_one() -> None:
    llm = {
        Workflow.PARSE_PASTED_PLAN,
        Workflow.EXTRACT_OFFER_EVIDENCE,
        Workflow.WRITE_JUSTIFICATIONS,
    }
    for name in llm:
        assert WORKFLOWS[name].queue is queue_for("openrouter")
    assert WORKFLOWS[Workflow.FETCH_PLACE_CANDIDATES].queue is Queue.DEFAULT


def test_inputs_accept_the_documented_fields() -> None:
    trip, doc = uuid4(), uuid4()
    parsed = ParsePastedPlanInput(
        trip_id=trip, document_id=doc, city_slug="berlin", provider="local"
    )
    assert parsed.provider == "local"
    offer = ExtractOfferEvidenceInput(
        trip_id=trip, document_id=doc, requirement_keys=["pool"]
    )
    assert offer.provider == "openrouter"
    assert WriteJustificationsInput(plan_id=uuid4()).provider == "openrouter"


def test_fetch_place_candidates_needs_exactly_one_city() -> None:
    assert FetchPlaceCandidatesInput(city_query="Gdańsk").city_slug is None
    assert FetchPlaceCandidatesInput(city_slug="gdansk").city_query is None
    with pytest.raises(ValueError, match="exactly one"):
        FetchPlaceCandidatesInput()
    with pytest.raises(ValueError, match="exactly one"):
        FetchPlaceCandidatesInput(city_query="a", city_slug="b")


def test_parse_output_leaves_room_for_the_matching_step() -> None:
    output = ParsePastedPlanOutput(items=[])
    assert output.matches == []
    assert output.unread == []
