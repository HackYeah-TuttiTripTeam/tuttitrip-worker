"""A decision agent under ``DBOSDurability``, resolved through the catalog."""

from dbos import DBOS
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.models.decision import UnfillableRoute
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

from tuttitrip_worker.shared.config.settings import Settings
from tuttitrip_worker.shared.llm.models import ModelKey, catalog, model_id

decision_agent = Agent(
    model_id(ModelKey.DECIDE),
    name="decision_check",
    defer_model_check=True,
    capabilities=[catalog.capability(), DBOSDurability()],
)


@DBOS.workflow(name="test_decision_workflow")
async def decision_workflow(question: str) -> str:
    result = await decision_agent.run(question, model=model_id(ModelKey.DECIDE))
    return result.output


def test_decision_agent_is_a_checkpointed_step_and_escalates(dbos: Settings) -> None:
    del dbos

    name = "basal"

    def basal(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        raise UnfillableRoute(name, "Reply", 0.9)

    def qwen(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("Qwen")])

    with catalog.override(FallbackModel(FunctionModel(basal), FunctionModel(qwen))):
        handle = DBOS.start_workflow(decision_workflow, "pytanie")
        assert handle.get_result() == "Qwen"
    steps = [s["function_name"] for s in DBOS.list_workflow_steps(handle.workflow_id)]
    assert "decision_check__model.request" in steps
