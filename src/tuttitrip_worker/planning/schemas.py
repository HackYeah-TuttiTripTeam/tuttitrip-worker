"""Internal payloads of the planning domain (pure: Pydantic only)."""

from pydantic import BaseModel, Field


class TripPlanDraft(BaseModel):
    """Structured output of the planner agent.

    Kept small on purpose: local models are unreliable with large outputs.
    """

    destination: str = Field(description="City or region of the trip.")
    days: int = Field(ge=1, le=60, description="Number of days.")
    highlights: list[str] = Field(
        max_length=10, description="Places or activities worth including."
    )


class SidedPerson(BaseModel):
    """A person on the "yes" or "no" side of a verdict, with the reason code."""

    name: str
    reason_code: str | None = None


class PersonUtility(BaseModel):
    """``explain()`` of one person for one place (E1): match, effort, utility."""

    name: str
    match: float
    effort: float
    utility: float


class VerdictFacts(BaseModel):
    """Everything the algorithm computed about the verdict on one place.

    This is the only data a justification may use: the model words it, the
    checker rejects any number or name that is not here.
    """

    place_id: str
    place_name: str | None = None
    verdict: str
    v_p: float | None = None
    yes: list[SidedPerson] = Field(default_factory=list)
    no: list[SidedPerson] = Field(default_factory=list)
    skip_codes: list[str] = Field(default_factory=list)
    substitute_name: str | None = None
    people: list[PersonUtility] = Field(default_factory=list)


class VerdictBatch(BaseModel):
    """One model call: a few verdicts, also the dependencies of the agent."""

    facts: list[VerdictFacts]


class JustificationItem(BaseModel):
    """Model output for one place: at most two short sentences."""

    place_id: str = Field(description="The place_id of the verdict, copied as is.")
    text: str = Field(description="At most two short sentences, using only given data.")


class JustificationDraft(BaseModel):
    """Structured output of the justifier agent: one item per verdict."""

    items: list[JustificationItem]
