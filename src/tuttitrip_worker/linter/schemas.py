"""Internal payloads of the linter domain (pure: Pydantic and dataclasses only)."""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from tuttitrip_worker.contracts import TransportMode

MAX_DRAFT_ITEMS = 300
"""Same cap as ``ParsePastedPlanOutput.items`` (items plus unread stay bounded)."""


@dataclass(frozen=True, slots=True)
class PastedText:
    """Run dependencies of the parser agent: the text the quotes must come from."""

    text: str


class DraftPlanItem(BaseModel):
    """One item as the model reads it; ``quote`` must be verbatim from the text.

    Deliberately lenient (no time or currency patterns): the strict
    ``ParsedPlanItem`` of the contract decides in ``logic/quotes.py`` and an
    item it rejects goes to ``unread`` instead of failing the whole run.
    """

    model_config = ConfigDict(frozen=True)

    day: int | None = Field(
        default=None, description="Day number (1 = first day) when the text says it."
    )
    start_time: str | None = Field(
        default=None, description="Start time as HH:MM, only if written in the text."
    )
    end_time: str | None = Field(
        default=None, description="End time as HH:MM, only if written in the text."
    )
    place_name: str = Field(description="Name of the place or activity, as written.")
    address: str | None = Field(default=None, description="Street address if given.")
    city: str | None = Field(default=None, description="City if given.")
    amount_minor: int | None = Field(
        default=None,
        description=(
            "Price PER PERSON in minor units (grosze, cents) only if the text "
            "states it. Leave empty for a group total or when not stated."
        ),
    )
    currency: str | None = Field(
        default=None, description="ISO 4217 code such as PLN or EUR."
    )
    transport: TransportMode | None = Field(
        default=None, description="Transport mode of the leg to this item, if stated."
    )
    quote: str = Field(
        description=(
            "A fragment copied character for character from the pasted text "
            "that supports this item (the line or sentence naming it)."
        )
    )


class PlanDraft(BaseModel):
    """Structured output of the parser agent: every item of the pasted plan."""

    items: list[DraftPlanItem] = Field(max_length=MAX_DRAFT_ITEMS)
