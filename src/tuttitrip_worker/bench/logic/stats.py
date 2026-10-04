"""Statistics of a run: latency percentiles, cost and the cells of the report (pure)."""

import math
from collections.abc import Iterable, Mapping

from tuttitrip_worker.bench.constants import (
    CALLS_PER_COST_UNIT,
    LANG_EN,
    LANG_PL,
    MEDIAN,
    P95,
    PASS_SCORE,
    TOKENS_PER_MILLION,
    Outcome,
    Price,
)
from tuttitrip_worker.bench.schemas import ExampleResult, Summary


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Nearest-rank percentile.

    Args:
        values: Samples.
        quantile: 0..1, for example 0.95.

    Returns:
        The sample at that rank, or ``None`` without samples.
    """
    ordered = sorted(values)
    if not ordered:
        return None
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def call_cost(
    prices: Mapping[str, Price], model_name: str, input_tokens: int, output_tokens: int
) -> float | None:
    """Cost of one call in USD.

    Args:
        prices: Token prices by model name.
        model_name: Name of the model that answered.
        input_tokens: Prompt tokens, retries included.
        output_tokens: Completion tokens, retries included.

    Returns:
        The cost, or ``None`` when the price is not known.
    """
    price = prices.get(model_name)
    if price is None:
        return None
    return (input_tokens * price[0] + output_tokens * price[1]) / TOKENS_PER_MILLION


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(case: str, model: str, results: list[ExampleResult]) -> Summary:
    """Aggregate the results of one model on one case.

    Accuracy is the mean score over answered examples (valid or invalid); a
    provider error says nothing about the model and is reported as its own rate.

    Args:
        case: Case name.
        model: Catalog route name.
        results: One result per example.

    Returns:
        The cell of the report.
    """
    total = len(results)
    if results and all(r.outcome is Outcome.SKIPPED for r in results):
        return Summary(
            case=case,
            model=model,
            total=total,
            answered=0,
            accuracy=None,
            passed=None,
            p50_seconds=None,
            p95_seconds=None,
            cost_per_100=None,
            invalid_rate=None,
            error_rate=None,
            skipped=True,
        )
    answered = [r for r in results if r.outcome in {Outcome.OK, Outcome.INVALID}]
    ran = [r for r in results if r.outcome is not Outcome.SKIPPED]
    latencies = [r.latency_seconds for r in answered if r.latency_seconds is not None]
    costs = [r.cost_usd for r in answered]
    known_costs = [c for c in costs if c is not None]
    return Summary(
        case=case,
        model=model,
        total=total,
        answered=len(answered),
        accuracy=_mean([r.score for r in answered]),
        passed=_mean([float(r.score >= PASS_SCORE) for r in answered]),
        p50_seconds=percentile(latencies, MEDIAN),
        p95_seconds=percentile(latencies, P95),
        cost_per_100=(
            sum(known_costs) / len(known_costs) * CALLS_PER_COST_UNIT
            if known_costs and len(known_costs) == len(costs)
            else None
        ),
        invalid_rate=(
            sum(r.outcome is Outcome.INVALID for r in answered) / len(answered)
            if answered
            else None
        ),
        error_rate=(
            sum(r.outcome is Outcome.ERROR for r in ran) / len(ran) if ran else None
        ),
        pl_accuracy=_mean([r.score for r in answered if r.lang == LANG_PL]),
        en_accuracy=_mean([r.score for r in answered if r.lang == LANG_EN]),
    )
