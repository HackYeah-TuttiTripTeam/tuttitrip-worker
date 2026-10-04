"""Fixed values of the benchmark."""

from enum import StrEnum
from typing import Final

EXAMPLES_FILE: Final = "examples.jsonl"
"""Golden examples of a case: one JSON object per line."""

RUBRIC_FILE: Final = "rubric.md"
"""Judge rubric: ``tests/golden/rubric.md`` is common, ``<case>/rubric.md`` adds."""

REPORT_PREFIX: Final = "llm-"
"""Report file name: ``llm-<date>.md``."""

ENV_GB10_KEY: Final = "GB10_LITELLM_KEY"
"""Variable of the env file that holds the GB10 gateway key."""

ENV_OPENROUTER_KEY: Final = "OPENROUTER_API_KEY"
"""Variable of the env file that holds the OpenRouter key."""

ENV_ANTHROPIC_KEY: Final = "ANTHROPIC_API_KEY"
"""Variable of the env file that holds the Anthropic key (judge)."""

ANTHROPIC_SLUG_PREFIX: Final = "anthropic/"
"""OpenRouter vendor prefix of the judge when it is called through OpenRouter."""

CALLS_PER_COST_UNIT: Final = 100
"""The report states the cost of this many calls."""

TOKENS_PER_MILLION: Final = 1_000_000
"""Prices are quoted per this many tokens."""

MEDIAN: Final = 0.5
"""Quantile of the p50 latency."""

P95: Final = 0.95
"""Quantile of the p95 latency."""

MIN_SCORE: Final = 0.0
"""Lowest score of an example."""

MAX_SCORE: Final = 1.0
"""Highest score of an example."""

PASS_SCORE: Final = 0.8
"""An example at or above this score counts as passed in the report."""

SEED_PREFIX: Final = "bench"
"""Prompt block seed of the benchmark (stable, so runs are comparable)."""

TRIP_ID: Final = "00000000-0000-4000-8000-000000000046"
"""Placeholder trip id the planner prompt carries."""

DEFAULT_MODELS: Final = (
    "agent",
    "chat",
    "decide",
    "decide-laya",
    "decide-cloud",
    "openrouter",
)
"""Catalog routes compared by default (the part after ``tuttitrip:``)."""

JUDGE_TEMPERATURE: Final = 0.0
"""The judge answers deterministically as far as the provider allows."""


class Outcome(StrEnum):
    """How one call of a model on one example ended."""

    OK = "ok"
    """A valid answer; it was scored."""
    INVALID = "invalid"
    """The model never produced a valid structured answer (scores 0)."""
    ERROR = "error"
    """The provider failed (network, quota, timeout); not the model's fault."""
    SKIPPED = "skipped"
    """The model cannot take this case (a decision model on free text) or has no key."""


class Role(StrEnum):
    """What kind of answer a case needs from a model."""

    GENERATIVE = "generative"
    """Free structured text: only language models can answer."""
    ANY = "any"
    """A pick-one answer: decision models and language models can answer."""


JUDGE_WEIGHT_OFFER: Final = 0.3
"""Share of the judge in an offer example (relevance of the quotes)."""

JUDGE_WEIGHT_PLAN: Final = 0.15
"""Share of the judge in a pasted-plan example (addresses and cities)."""

JUDGE_WEIGHT_TRIP: Final = 0.6
"""Share of the judge in a generated plan (the highlights fit the request)."""

JUDGE_WEIGHT_EXPENSE: Final = 0.1
"""Share of the judge in an expense example (the description)."""

JUDGE_WEIGHT_RECEIPT: Final = 0.15
"""Share of the judge in a receipt example (the item names)."""

MAX_HIGHLIGHTS: Final = 10
"""Most highlights the planner schema allows."""

NO_PLACE: Final = None
"""Reference answer of a match example whose item is not in the catalog."""

STATE_UNCONFIRMED: Final = "unconfirmed"
"""Requirement state: the offer is silent (no quotes)."""

STATE_CONFLICTING: Final = "conflicting"
"""Requirement state: quotes say both present and absent."""

VERDICT_PRESENT: Final = "present"
"""Quote verdict: the offer provides what the requirement asks for."""

VERDICT_ABSENT: Final = "absent"
"""Quote verdict: the offer denies it or offers something else."""

VERDICT_NOT_APPLICABLE: Final = "not_applicable"
"""Quote verdict: the quote does not settle the requirement."""

type Price = tuple[float, float] | None
"""USD per million tokens, ``(input, output)``; ``None`` = not published."""

PRICES: Final[dict[str, Price]] = {
    # Own hardware (GB10): no per-token price. Electricity and depreciation are
    # not counted, which the report says.
    "qwen3.8-27b": (0.0, 0.0),
    "qwen3.8-27b-chat": (0.0, 0.0),
    "basal": (0.0, 0.0),
    "laya": (0.0, 0.0),
    # OpenRouter list prices of 2026-10-04 (https://openrouter.ai/models).
    "google/gemini-3.8-flash": (0.75, 3.75),
    "anthropic/claude-sonnet-5.5": (2.0, 10.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    # JEV 1.13 is not listed with a price on OpenRouter.
    "typesafe/jev-1.13": None,
}
"""Token prices by model name, for the cost column of the report."""

ROUTE_GB10: Final = frozenset({"agent", "chat", "decide", "decide-laya"})
"""Routes whose first model needs the GB10 gateway key."""

ROUTE_OPENROUTER: Final = frozenset({"openrouter", "decide-cloud"})
"""Routes whose first model needs the OpenRouter key."""

ROUTE_DECISION: Final = frozenset({"decide", "decide-laya", "decide-cloud"})
"""Routes served by decision models (pick-one answers only)."""

JUDGE_RETRIES: Final = 2
"""Extra tries of the judge when its answer is not valid."""

ERROR_PREVIEW_CHARS: Final = 200
"""Longest error text kept in a result."""

SLUG_SEPARATOR: Final = "/"
"""Separates vendor and model in an OpenRouter slug (``anthropic/claude-...``)."""

ASSIGN: Final = "="
"""Separates name and value in an env file."""

QUOTE_CHARS: Final = frozenset({'"', "'"})
"""Quote characters that may surround an env file value."""

LANG_PL: Final = "pl"
"""Language tag of Polish examples."""

LANG_EN: Final = "en"
"""Language tag of English examples."""

ALTERNATIVE_SEPARATOR: Final = "|"
"""Separates acceptable destination names in a golden example."""
