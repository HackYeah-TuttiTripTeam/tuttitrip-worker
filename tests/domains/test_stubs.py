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
    ParsedPlanItem,
    ParsePastedPlanInput,
    ParsePastedPlanOutput,
    PlaceMatch,
    Queue,
    RequirementLabel,
    Workflow,
    WriteJustificationsInput,
    queue_for,
)
from tuttitrip_worker.shared.config.settings import Settings

STUBS: dict[Workflow, dict[str, Any]] = {
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
    payload = {"contract_version": CONTRACT_VERSION}  # no plan id given
    handle = enqueue(client, dbos, Workflow.WRITE_JUSTIFICATIONS, payload)
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


def test_offer_requirement_keys_must_be_unique_and_labelled_keys_known() -> None:
    trip, doc = uuid4(), uuid4()
    with pytest.raises(ValueError, match="unique"):
        ExtractOfferEvidenceInput(
            trip_id=trip, document_id=doc, requirement_keys=["pool", "pool"]
        )
    with pytest.raises(ValueError, match="requirement_keys"):
        ExtractOfferEvidenceInput(
            trip_id=trip,
            document_id=doc,
            requirement_keys=["pool"],
            requirements=[RequirementLabel(key="spa", label="Spa")],
        )
    ok = ExtractOfferEvidenceInput(
        trip_id=trip,
        document_id=doc,
        requirement_keys=["pool"],
        requirements=[RequirementLabel(key="pool", label="Basen")],
    )
    assert ok.requirements is not None


def test_parsed_item_normalizes_times_and_validates_currency() -> None:
    item = ParsedPlanItem(
        index=0, start_time="9:00", end_time="10:30", place_name="Wawel", quote="x"
    )
    assert (item.start_time, item.end_time, item.day) == ("09:00", "10:30", None)
    with pytest.raises(ValueError, match="currency"):
        ParsedPlanItem(index=0, place_name="a", quote="x", currency="pln")
    with pytest.raises(ValueError, match="start_time"):
        ParsedPlanItem(index=0, place_name="a", quote="x", start_time="25:00")


def test_place_match_needs_an_explicit_status() -> None:
    assert PlaceMatch(item_index=0, status="unrecognized").place_id is None
    with pytest.raises(ValueError, match="status"):
        PlaceMatch.model_validate({"item_index": 0})


def test_slugs_follow_the_shared_pattern() -> None:
    assert FetchPlaceCandidatesInput(city_slug="gdansk-polska").city_slug
    with pytest.raises(ValueError, match="city_slug"):
        FetchPlaceCandidatesInput(city_slug="Gdańsk")
