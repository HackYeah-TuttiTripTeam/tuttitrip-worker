"""Payloads of the benchmark (pure: Pydantic only)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tuttitrip_worker.bench.constants import MAX_SCORE, MIN_SCORE, Outcome

type Json = dict[str, Any]


class Example(BaseModel):
    """One line of a golden set: an input, the reference answer and notes."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    lang: Literal["pl", "en"]
    tags: list[str] = Field(default_factory=list)
    input: Json
    expected: Json
    notes: str = ""


class JudgeVerdict(BaseModel):
    """The judge's score of one answer against the reference."""

    score: float = Field(
        ge=MIN_SCORE,
        le=MAX_SCORE,
        description=(
            "0 to 1 by the rubric: 1 = as good as the reference, 0 = wrong or "
            "invented. Use the steps 0, 0.25, 0.5, 0.75 and 1."
        ),
    )
    reason: str = Field(
        max_length=300, description="One or two sentences naming the deciding facts."
    )


class JudgeView(BaseModel):
    """What the judge sees: only the parts that code cannot check."""

    model_config = ConfigDict(frozen=True)

    criteria: str
    reference: Json
    candidate: Json


class Scored(BaseModel):
    """Result of the deterministic checks of one answer.

    ``deterministic`` is the mean of ``parts`` (each 0..1). When ``judge`` is
    set, the judge scores that view and ``judge_weight`` is its share of the
    final score; the rest is ``deterministic``.
    """

    model_config = ConfigDict(frozen=True)

    deterministic: float = Field(ge=MIN_SCORE, le=MAX_SCORE)
    parts: dict[str, float] = Field(default_factory=dict)
    judge: JudgeView | None = None
    judge_weight: float = Field(default=0.0, ge=MIN_SCORE, le=MAX_SCORE)


class ExampleResult(BaseModel):
    """One model on one example."""

    model_config = ConfigDict(frozen=True)

    case: str
    model: str
    example_id: str
    lang: Literal["pl", "en"]
    outcome: Outcome
    score: float = Field(default=MIN_SCORE, ge=MIN_SCORE, le=MAX_SCORE)
    deterministic: float | None = None
    judge_score: float | None = None
    parts: dict[str, float] = Field(default_factory=dict)
    reason: str = ""
    latency_seconds: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    output: Json | None = None
    error: str | None = None


class Summary(BaseModel):
    """One cell of the report: a model on a case."""

    model_config = ConfigDict(frozen=True)

    case: str
    model: str
    total: int
    answered: int
    accuracy: float | None
    passed: float | None
    p50_seconds: float | None
    p95_seconds: float | None
    cost_per_100: float | None
    invalid_rate: float | None
    error_rate: float | None
    skipped: bool = False
    pl_accuracy: float | None = None
    en_accuracy: float | None = None
