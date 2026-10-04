"""Run cases on models: calls, timing, scoring and the judge."""

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path

from tuttitrip_worker.bench.agents import (
    JudgeSetup,
    Route,
    Subject,
    classify,
    judge,
    subject_for,
)
from tuttitrip_worker.bench.cases import Case
from tuttitrip_worker.bench.constants import MIN_SCORE, PRICES, Outcome, Role
from tuttitrip_worker.bench.errors import CaseUnavailableError
from tuttitrip_worker.bench.logic.scoring import combine
from tuttitrip_worker.bench.logic.stats import call_cost
from tuttitrip_worker.bench.schemas import Example, ExampleResult

JUDGE_FAILED = "judge unavailable; the score is the code checks only"
"""Reason kept when the judge could not answer."""


@dataclass(frozen=True)
class Target:
    """What the runner needs besides the model: the case and where its files are."""

    case: Case
    golden: Path
    """Directory of the case's golden set (images live there)."""
    judge: JudgeSetup | None
    rubric: str
    timeout: float
    """Longest wait for one example, retries included."""


@dataclass
class JudgeLedger:
    """Tokens the judge used in a run (its cost is stated in the report)."""

    input_tokens: int = 0
    output_tokens: int = 0


def _result(
    target: Target, route: Route, example: Example, outcome: Outcome, **fields: object
) -> ExampleResult:
    return ExampleResult.model_validate(
        {
            "case": target.case.name,
            "model": route.name,
            "example_id": example.id,
            "lang": example.lang,
            "outcome": outcome,
            **fields,
        }
    )


def _usage(subject: Subject) -> dict[str, object]:
    meter = subject.model
    return {
        "input_tokens": meter.input_tokens,
        "output_tokens": meter.output_tokens,
        "cost_usd": call_cost(
            PRICES, meter.model_name, meter.input_tokens, meter.output_tokens
        ),
    }


async def evaluate(
    target: Target, route: Route, example: Example, ledger: JudgeLedger
) -> ExampleResult:
    """Run one model on one example and score the answer.

    Args:
        target: The case, its files, the judge and the timeout.
        route: The model under test.
        example: Golden example.
        ledger: Collects the judge's token use.

    Returns:
        The result; never raises for a failing model.
    """
    if route.model is None:
        return _result(
            target, route, example, Outcome.SKIPPED, error="no API key for this route"
        )
    if target.case.role is Role.GENERATIVE and route.decision:
        return _result(
            target,
            route,
            example,
            Outcome.SKIPPED,
            error="a decision model cannot write free text",
        )
    subject = subject_for(route)
    started = time.monotonic()
    try:
        async with asyncio.timeout(target.timeout):
            answer = await target.case.run(example, subject, target.golden)
    except CaseUnavailableError as error:
        return _result(target, route, example, Outcome.SKIPPED, error=str(error))
    except Exception as error:  # ruff: ignore[blind-except] - every failure becomes a result
        outcome, text = classify(error)
        return _result(
            target,
            route,
            example,
            outcome,
            score=MIN_SCORE,
            latency_seconds=time.monotonic() - started,
            error=text,
            **_usage(subject),
        )
    elapsed = time.monotonic() - started
    scored = target.case.score(example, answer)
    judge_score, reason = None, ""
    if target.judge is not None and scored.judge is not None:
        verdict = await judge(target.judge, target.rubric, scored.judge, example.input)
        if verdict is None:
            reason = JUDGE_FAILED
        else:
            judge_score, reason = verdict[0].score, verdict[0].reason
            ledger.input_tokens += verdict[1].input_tokens
            ledger.output_tokens += verdict[1].output_tokens
    return _result(
        target,
        route,
        example,
        Outcome.OK,
        score=combine(scored, judge_score),
        deterministic=scored.deterministic,
        judge_score=judge_score,
        parts=scored.parts,
        reason=reason,
        latency_seconds=elapsed,
        output=answer,
        **_usage(subject),
    )


async def run_route(
    target: Target,
    route: Route,
    examples: list[Example],
    ledger: JudgeLedger,
    concurrency: int,
) -> list[ExampleResult]:
    """Run one model on all examples of a case, a few at a time.

    Args:
        target: The case, its files, the judge and the timeout.
        route: The model under test.
        examples: Golden examples.
        ledger: Collects the judge's token use.
        concurrency: Parallel calls to this model.

    Returns:
        One result per example, in order.
    """
    gate = asyncio.Semaphore(concurrency)

    async def one(example: Example) -> ExampleResult:
        async with gate:
            return await evaluate(target, route, example, ledger)

    return list(await asyncio.gather(*(one(example) for example in examples)))
