"""Durable Pydantic AI agents of the planning domain.

Agents are module-level so they exist before ``DBOS.launch()``. Each has a
unique ``name`` (it prefixes its DBOS step names) and the ``DBOSDurability``
capability; a run is durable only when ``agent.run()`` is called inside a
``@DBOS.workflow``. Models are chosen per run with ``model=model_id(...)``.
"""

import logging

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.settings import ModelSettings

from tuttitrip_worker.contracts import APPLICATION_NAME, LlmProvider
from tuttitrip_worker.planning.constants import (
    JUSTIFICATION_RETRIES,
    JUSTIFICATION_TEMPERATURE,
    LANGUAGE_NAMES,
)
from tuttitrip_worker.planning.logic.justification_check import check_batch, passing
from tuttitrip_worker.planning.schemas import (
    JustificationDraft,
    JustificationItem,
    TripPlanDraft,
    VerdictBatch,
)
from tuttitrip_worker.shared.llm.models import catalog, model_id

logger = logging.getLogger(APPLICATION_NAME)

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


verdict_justifier: Agent[VerdictBatch, JustificationDraft] = Agent(
    "tuttitrip:chat",
    name="verdict_justifier",
    deps_type=VerdictBatch,
    output_type=JustificationDraft,
    retries={"output": JUSTIFICATION_RETRIES},
    model_settings=ModelSettings(temperature=JUSTIFICATION_TEMPERATURE),
    instructions=(
        "You word the verdict an algorithm gave on each place of a family or "
        "group trip. The user message is JSON: for every place the verdict "
        "(must, fits, iconic_not_yours or skip), the weighted opinion v_p, who "
        "is for (yes) and who is against (no) with reason codes, the rejection "
        "codes of a skip (veto, blocked, closed, no_fit, segment, stairs), a "
        "substitute place, and per person the match, effort and utility. "
        "Write for each place at most two short sentences that say why, by "
        "naming the people on each side and the reasons. A veto must be "
        "named as a veto. Use ONLY the names, numbers and facts in the JSON: "
        "copy a number as given (a fraction may be written as a percent), "
        "never compute, round differently, add prices, opening hours, "
        "distances or any other fact. Skip numbers when they add nothing. "
        "Return one item per place with its place_id copied unchanged."
    ),
    defer_model_check=True,
    capabilities=[catalog.capability(), DBOSDurability()],
)


@verdict_justifier.output_validator
def _texts_must_come_from_the_data(
    ctx: RunContext[VerdictBatch], draft: JustificationDraft
) -> JustificationDraft:
    """Send the model back when a text states a number or a name not in the data.

    On the last attempt the draft passes unchanged: ``write_batch`` keeps only
    the entries that pass the checks, the others get the backend's template.

    Args:
        ctx: Run context; ``deps`` carries the verdict facts of the batch.
        draft: The model's structured output.

    Returns:
        The draft.

    Raises:
        ModelRetry: An entry fails its checks (not the last try).
    """
    if ctx.last_attempt:
        return draft
    problems = check_batch(draft.items, ctx.deps.facts)
    if problems:
        msg = "Fix these justifications:\n" + "\n".join(problems)
        raise ModelRetry(msg)
    return draft


def _prompt(batch: VerdictBatch, locale: str) -> str:
    return (
        f"Write the justifications in {LANGUAGE_NAMES[locale]}.\n"
        f"{batch.model_dump_json(exclude_none=True)}"
    )


async def write_batch(batch: VerdictBatch, locale: str) -> list[JustificationItem]:
    """Justify a batch of verdicts (call it inside the workflow to be durable).

    Args:
        batch: Verdict facts of a few places.
        locale: ``pl`` or ``en``.

    Returns:
        The entries that pass every check; a place that is missing has none.
        When the model never gives a usable answer, nothing is returned.
    """
    try:
        result = await verdict_justifier.run(_prompt(batch, locale), deps=batch)
    except UnexpectedModelBehavior:
        logger.warning("justifier gave no valid answer for %d places", len(batch.facts))
        return []
    return passing(result.output.items, batch.facts)
