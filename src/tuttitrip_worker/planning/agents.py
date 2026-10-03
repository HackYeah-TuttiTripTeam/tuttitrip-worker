"""Durable Pydantic AI agents of the planning domain.

Agents are module-level so they exist before ``DBOS.launch()``. Each has a
unique ``name`` (it prefixes its DBOS step names) and the ``DBOSDurability``
capability; a run is durable only when ``agent.run()`` is called inside a
``@DBOS.workflow``. Models are chosen per run with ``model=model_id(...)``.
"""

from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSDurability

from tuttitrip_worker.contracts import LlmProvider
from tuttitrip_worker.planning.schemas import TripPlanDraft
from tuttitrip_worker.shared.llm.models import catalog, model_id

planner_agent = Agent(
    model_id(LlmProvider.OPENROUTER),
    name="trip_planner",
    output_type=TripPlanDraft,
    instructions=(
        "You draft short trip plans for families and groups. Answer with the "
        "destination, the number of days and up to ten highlights. Do not "
        "invent prices or opening hours; deterministic checks run later."
    ),
    defer_model_check=True,
    capabilities=[catalog.capability(), DBOSDurability()],
)
