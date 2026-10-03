"""Internal payloads of the accommodation domain (pure: Pydantic only)."""

from pydantic import BaseModel, Field


class RequirementQuotes(BaseModel):
    """Quotes the extractor found for one requirement (none = offer is silent)."""

    requirement_key: str = Field(description="Key of the requirement, copied as given.")
    quotes: list[str] = Field(
        default_factory=list,
        description=(
            "Short passages copied character for character from the offer that "
            "say something about this requirement. Empty when the offer does "
            "not mention it."
        ),
    )


class ExtractedQuotes(BaseModel):
    """Structured output of the quote extractor: one entry per requirement."""

    requirements: list[RequirementQuotes] = Field(default_factory=list)
