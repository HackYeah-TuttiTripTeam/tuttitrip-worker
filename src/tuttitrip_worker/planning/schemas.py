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
