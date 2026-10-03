"""Planning domain: the durable agent run, with TestModel/FunctionModel only."""

from typing import Any
from uuid import uuid4

import pytest
from dbos import DBOS, DBOSClient
from pydantic_ai import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from tests.helpers import enqueue
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    GenerateTripPlanOutput,
    LlmProvider,
    Workflow,
)
from tuttitrip_worker.planning.agents import planner_agent
from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.db import job_results
from tuttitrip_worker.shared.llm.models import catalog, model_id

PLAN = {"destination": "Kraków", "days": 3, "highlights": ["Wawel", "Kazimierz"]}


@pytest.fixture
def saved(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, dict[str, Any]]]:
    calls: list[tuple[str, str, dict[str, Any]]] = []

    async def fake_save(workflow_id: str, name: str, result: dict[str, Any]) -> None:
        calls.append((workflow_id, name, result))

    monkeypatch.setattr(job_results, "save_job_result", fake_save)
    return calls


@pytest.mark.parametrize("provider", list(LlmProvider))
def test_generate_trip_plan_runs_the_agent_durably(
    client: DBOSClient,
    dbos: Settings,
    saved: list[tuple[str, str, dict[str, Any]]],
    provider: LlmProvider,
) -> None:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "trip_id": str(uuid4()),
        "request": "Weekend w Krakowie z dziećmi",
        "provider": provider.value,
    }
    with catalog.override(TestModel(custom_output_args=PLAN)):
        handle = enqueue(client, dbos, Workflow.GENERATE_TRIP_PLAN, payload)
        output = GenerateTripPlanOutput.model_validate(handle.get_result())

    assert output.destination == "Kraków"
    assert output.highlights == ["Wawel", "Kazimierz"]
    workflow_id = handle.get_workflow_id()
    assert saved == [
        (workflow_id, "generate_trip_plan", output.model_dump(mode="json"))
    ]
    assert client.get_event(workflow_id, "progress") == {
        "stage": "done",
        "percent": 100,
    }
    steps = [step["function_name"] for step in DBOS.list_workflow_steps(workflow_id)]
    # DBOSDurability turned the model request into a checkpointed step.
    assert "trip_planner__model.request" in steps


def test_agent_outside_a_workflow_is_a_normal_agent() -> None:
    prompts: list[str] = []

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompts.append(str(messages[-1].parts[-1]))
        output_tool = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(output_tool, PLAN)])

    with catalog.override(FunctionModel(answer)):
        result = planner_agent.run_sync("Gdańsk", model=model_id(LlmProvider.LOCAL))
    assert result.output.destination == "Kraków"
    assert "Gdańsk" in prompts[0]
