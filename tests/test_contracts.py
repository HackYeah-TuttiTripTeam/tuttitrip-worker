"""The contract: rendered document, versioning, and registration in DBOS."""

from pathlib import Path
from typing import Any

import pytest
from dbos import DBOSClient, PortableWorkflowError

from tests.helpers import enqueue
from tuttitrip_worker import main as worker_main
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    WORKFLOWS,
    ContractError,
    ErrorCode,
    LlmProvider,
    PingInput,
    Queue,
    Workflow,
    contract_document,
    contract_json,
    parse_input,
    queue_for,
)
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.dbos.runtime import QUEUE_LIMITS

SCHEMA_FILE = Path(__file__).resolve().parents[1] / "contracts" / "jobs.schema.json"


def test_committed_contract_file_is_up_to_date() -> None:
    assert SCHEMA_FILE.read_text(encoding="utf-8") == contract_json(), (
        "run `uv run python scripts/export_contracts.py` and commit the result"
    )


def test_document_has_no_docs_that_could_cause_drift() -> None:
    rendered = contract_json()
    assert '"title"' not in rendered
    assert '"description"' not in rendered
    assert set(contract_document()) == {
        "contract_version",
        "application_name",
        "queues",
        "workflows",
        "events",
    }


def test_current_version_is_supported() -> None:
    assert CONTRACT_VERSION in SUPPORTED_CONTRACT_VERSIONS


def test_every_contract_workflow_is_wired_in_main() -> None:
    assert set(worker_main.WORKFLOWS) == set(Workflow) == set(WORKFLOWS)


def test_every_queue_has_limits() -> None:
    assert set(QUEUE_LIMITS) == set(Queue)


def test_llm_jobs_go_to_the_provider_queue() -> None:
    assert queue_for(LlmProvider.LOCAL) is Queue.LOCAL_LLM
    assert queue_for("openrouter") is Queue.OPENROUTER


@pytest.mark.parametrize("payload", [{}, {"contract_version": 999}, "nope", None])
def test_unsupported_versions_are_rejected(payload: object) -> None:
    with pytest.raises(ContractError) as info:
        parse_input(PingInput, payload)
    assert info.value.code == ErrorCode.UNSUPPORTED_CONTRACT_VERSION.value
    assert info.value.data["supported_versions"] == sorted(SUPPORTED_CONTRACT_VERSIONS)


def test_invalid_payloads_are_rejected_with_details() -> None:
    payload = {"contract_version": CONTRACT_VERSION, "message": "x" * 201}
    with pytest.raises(ContractError) as info:
        parse_input(PingInput, payload)
    assert info.value.code == ErrorCode.INVALID_PAYLOAD.value
    assert [tuple(e["loc"]) for e in info.value.data["errors"]] == [("message",)]


def test_unknown_fields_are_ignored_for_additive_changes() -> None:
    payload = {"contract_version": CONTRACT_VERSION, "message": "a", "new": 1}
    assert parse_input(PingInput, payload).message == "a"


@pytest.mark.parametrize("name", list(Workflow))
def test_workflow_is_registered_in_dbos_under_its_contract_name(
    client: DBOSClient, dbos: Settings, name: Workflow
) -> None:
    # Enqueued by name through DBOSClient, like the backend. Only our own
    # function raises ContractError, so this proves the name -> function
    # registration, the queue, the version gate and portable serialization.
    payload: dict[str, Any] = {"contract_version": 999}
    handle = enqueue(client, dbos, name, payload)
    with pytest.raises(PortableWorkflowError) as info:
        handle.get_result()
    assert info.value.name == "ContractError"
    assert info.value.code == ErrorCode.UNSUPPORTED_CONTRACT_VERSION.value
    assert isinstance(info.value.data, dict)
    assert info.value.data["received_version"] == 999
