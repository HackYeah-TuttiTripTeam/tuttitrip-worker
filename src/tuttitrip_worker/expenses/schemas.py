"""Internal payloads of the expenses domain (pure: Pydantic and dataclasses only)."""

from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from tuttitrip_worker.contracts import ExpenseCategoryName

MAX_RECEIPT_LINES = 100


@dataclass(frozen=True, slots=True)
class TypedExpense:
    """Run dependencies of the text agent: the sentence the amount must come from."""

    text: str


@dataclass(frozen=True, slots=True)
class TripDates:
    """When the trip runs (either end may be unknown)."""

    start: date | None = None
    end: date | None = None


class ExpenseTextReading(BaseModel):
    """What the model read from the sentence; checked by ``logic.amounts``."""

    model_config = ConfigDict(frozen=True)

    amount_minor: int | None = Field(
        default=None,
        description=(
            "Amount in minor units (grosze, cents), only if the text states "
            "it. 120,50 zl is 12050. Leave empty when there is no amount."
        ),
    )
    amount_text: str = Field(
        default="",
        description=(
            "The amount exactly as written in the text, for example '120,50 zl'."
        ),
    )
    currency: str | None = Field(
        default=None,
        description="ISO 4217 code, only if the text names it (zl/PLN, euro/EUR).",
    )
    description: str = Field(
        default="", description="Short note about what was bought, from the text."
    )
    payer_name: str | None = Field(
        default=None,
        description=(
            "Who paid, as written. Leave empty when the writer paid "
            "('I paid', 'zaplacilem')."
        ),
    )
    included_names: list[str] = Field(
        default_factory=list,
        description="People the text says shared the cost, as written.",
    )
    excluded_names: list[str] = Field(
        default_factory=list,
        description="People the text says did not take part, as written.",
    )


class ReceiptLine(BaseModel):
    """One line of a receipt as read from the image."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="", description="Item name as printed.")
    amount_minor: int = Field(description="Line total in minor units (grosze, cents).")


class ReceiptReading(BaseModel):
    """What the vision model read from a receipt or a bank screenshot."""

    model_config = ConfigDict(frozen=True)

    amount_minor: int | None = Field(
        default=None,
        description=(
            "Total paid in minor units (grosze, cents), only if visible. "
            "142,00 is 14200. Leave empty when it cannot be read."
        ),
    )
    currency: str | None = Field(
        default=None, description="ISO 4217 code, only if shown (zl/PLN, EUR)."
    )
    spent_on: str | None = Field(
        default=None, description="Date of the purchase as YYYY-MM-DD, if shown."
    )
    merchant: str | None = Field(default=None, description="Shop or payee name.")
    items: list[ReceiptLine] = Field(
        default_factory=list,
        max_length=MAX_RECEIPT_LINES,
        description="Printed lines with their totals, if the image lists them.",
    )


class CategoryDecision(BaseModel):
    """Decision about a reading: the spending category and whether to ask."""

    category: ExpenseCategoryName = Field(
        description=(
            "What the money was spent on: food (restaurants, groceries), "
            "transport, lodging, activities (tickets, tours), shopping or other."
        )
    )
    needs_confirmation: bool = Field(
        description="True when the reading looks unreliable or unusual."
    )


@dataclass(frozen=True, slots=True)
class Judgement:
    """A decision with the model's confidence (``None`` = it reported none)."""

    category: ExpenseCategoryName | None
    needs_confirmation: bool
    confidence: float | None
