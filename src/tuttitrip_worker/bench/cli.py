"""Command line of the benchmark.

    uv run python -m tuttitrip_worker.bench --case parse_expense_text \
        --models chat,openrouter --env-file ~/.config/tuttitrip/keys.env

Manual only: it calls real models and costs money, so CI never runs it. Keys
are read from the environment or from ``--env-file`` inside this process and
are never printed.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import SecretStr

from tuttitrip_worker.bench.agents import (
    JudgeSetup,
    Route,
    build_judge_model,
    first_leg,
)
from tuttitrip_worker.bench.cases import CASES, Case
from tuttitrip_worker.bench.constants import (
    DEFAULT_MODELS,
    ENV_ANTHROPIC_KEY,
    ENV_GB10_KEY,
    ENV_OPENROUTER_KEY,
    PRICES,
    REPORT_PREFIX,
    ROUTE_GB10,
    ROUTE_OPENROUTER,
    TOKENS_PER_MILLION,
)
from tuttitrip_worker.bench.golden import load_examples, load_rubric
from tuttitrip_worker.bench.logic.env import parse_env
from tuttitrip_worker.bench.logic.report import NO_DATA, RunInfo, render
from tuttitrip_worker.bench.logic.stats import summarize
from tuttitrip_worker.bench.runner import JudgeLedger, Target, run_route
from tuttitrip_worker.bench.schemas import Example, ExampleResult, Summary
from tuttitrip_worker.shared.config.settings import (
    BenchSettings,
    LlmSettings,
    get_settings,
)
from tuttitrip_worker.shared.llm.models import ModelKey

REPO_ROOT = Path(__file__).resolve().parents[3]
"""Checkout root: the golden sets and the reports live next to ``src``."""

QUIET_ENV = "PYDANTIC_AI_NO_BANNER"
"""Pydantic AI prints a banner on the first run unless this is set."""

ALL_CASES = "all"
"""Value of ``--case`` that selects every case."""


def _say(text: str) -> None:
    sys.stdout.write(f"{text}\n")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser.

    Returns:
        The parser of ``python -m tuttitrip_worker.bench``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m tuttitrip_worker.bench", description=__doc__
    )
    parser.add_argument(
        "--case",
        default=ALL_CASES,
        help=f"comma-separated cases or '{ALL_CASES}': {', '.join(CASES)}",
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="comma-separated catalog routes (the part after 'tuttitrip:')",
    )
    parser.add_argument("--env-file", type=Path, help="file with the API keys")
    parser.add_argument("--golden-dir", type=Path, default=REPO_ROOT / "tests/golden")
    parser.add_argument("--limit", type=int, help="only the first N examples per case")
    parser.add_argument("--concurrency", type=int, help="parallel calls per model")
    parser.add_argument("--judge", help="judge model (default: from settings)")
    parser.add_argument(
        "--no-judge", action="store_true", help="code checks only, no LLM judge"
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Markdown report (default: docs/benchmarks/llm-<date>.md)",
    )
    parser.add_argument(
        "--no-report", action="store_true", help="print the summary only"
    )
    return parser


def _secret(file_env: Mapping[str, str], name: str, current: SecretStr) -> SecretStr:
    """The configured secret, else the one from the env file, else the process env.

    Args:
        file_env: Variables of ``--env-file``.
        name: Variable name.
        current: Value from settings (may be empty).

    Returns:
        The secret to use; empty when none is set anywhere.
    """
    if current.get_secret_value():
        return current
    return SecretStr(file_env.get(name) or os.environ.get(name, ""))


def resolve_settings(
    file_env: Mapping[str, str],
) -> tuple[LlmSettings, BenchSettings]:
    """Settings with the keys filled in from the env file or the environment.

    Args:
        file_env: Variables of ``--env-file`` (they do not enter ``os.environ``).

    Returns:
        LLM settings and benchmark settings.
    """
    settings = get_settings()
    llm = settings.llm.model_copy(
        update={
            "gb10_api_key": _secret(file_env, ENV_GB10_KEY, settings.llm.gb10_api_key),
            "openrouter_api_key": _secret(
                file_env, ENV_OPENROUTER_KEY, settings.llm.openrouter_api_key
            ),
        }
    )
    bench = settings.bench.model_copy(
        update={
            "anthropic_api_key": _secret(
                file_env, ENV_ANTHROPIC_KEY, settings.bench.anthropic_api_key
            )
        }
    )
    return llm, bench


def build_routes(names: Sequence[str], llm: LlmSettings) -> list[Route]:
    """The first model of each requested route; ``None`` when its key is missing.

    Args:
        names: Catalog route names.
        llm: LLM settings with the keys.

    Returns:
        One route per name, in order.

    Raises:
        SystemExit: A name is not a catalog route.
    """
    have_gb10 = bool(llm.gb10_api_key.get_secret_value())
    have_openrouter = bool(llm.openrouter_api_key.get_secret_value())
    routes: list[Route] = []
    for name in names:
        try:
            key = ModelKey(name)
        except ValueError as error:
            msg = (
                f"unknown route {name!r}; known: {', '.join(m.value for m in ModelKey)}"
            )
            raise SystemExit(msg) from error
        usable = (name in ROUTE_GB10 and have_gb10) or (
            name in ROUTE_OPENROUTER and have_openrouter
        )
        routes.append(Route(name, first_leg(key, llm) if usable else None))
    return routes


def _price_text(route: Route) -> str:
    if route.model is None:
        return NO_DATA
    price = PRICES.get(route.model.model_name)
    if price is None:
        return f"{NO_DATA} (cena nieopublikowana)"
    return f"{price[0]:.2f} / {price[1]:.2f} USD"


def select_cases(spec: str) -> list[Case]:
    """Cases named on the command line.

    Args:
        spec: Comma-separated case names or ``all``.

    Returns:
        The cases in the order given.

    Raises:
        SystemExit: A name is not a case.
    """
    names = list(CASES) if spec == ALL_CASES else [n.strip() for n in spec.split(",")]
    unknown = [name for name in names if name not in CASES]
    if unknown:
        msg = f"unknown case {', '.join(unknown)}; known: {', '.join(CASES)}"
        raise SystemExit(msg)
    return [CASES[name] for name in names]


def examples_of(case: Case, golden_dir: Path, limit: int | None) -> list[Example]:
    """Golden examples of a case (derived ones come from their source).

    Args:
        case: The case.
        golden_dir: ``tests/golden``.
        limit: Keep only the first N examples.

    Returns:
        The examples.
    """
    examples = load_examples(golden_dir, case.source or case.name)
    if case.derive is not None:
        examples = case.derive(examples)
    return examples[:limit] if limit else examples


def _target(
    case: Case, golden_dir: Path, judge_setup: JudgeSetup | None, bench: BenchSettings
) -> Target:
    source = case.source or case.name
    return Target(
        case=case,
        golden=golden_dir / source,
        judge=judge_setup,
        rubric=load_rubric(golden_dir, source),
        timeout=bench.request_timeout_seconds,
    )


def _run_info(
    cases: Sequence[Case],
    routes: Sequence[Route],
    counts: Mapping[str, int],
    judge_setup: JudgeSetup | None,
    ledger: JudgeLedger,
) -> RunInfo:
    return RunInfo(
        day=datetime.now(tz=UTC).date(),
        cases=[case.name for case in cases],
        models={
            route.name: route.model.model_name if route.model else "brak klucza"
            for route in routes
        },
        judge=judge_setup.name if judge_setup else None,
        examples=counts,
        prices={route.name: _price_text(route) for route in routes},
        judge_cost=_judge_cost(judge_setup, ledger),
    )


def _write_report(
    path: Path | None,
    summaries: Sequence[Summary],
    results: Sequence[ExampleResult],
    info: RunInfo,
) -> None:
    target = path or (
        REPO_ROOT / "docs/benchmarks" / f"{REPORT_PREFIX}{info.day.isoformat()}.md"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(summaries, info), encoding="utf-8")
    target.with_suffix(".jsonl").write_text(
        "".join(f"{r.model_dump_json()}\n" for r in results), encoding="utf-8"
    )
    _say(f"Report: {target}")


async def _read_env(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    return parse_env(await asyncio.to_thread(path.read_text, encoding="utf-8"))


async def run(args: argparse.Namespace) -> int:
    """Run the selected cases on the selected routes and write the report.

    Args:
        args: Parsed arguments.

    Returns:
        Process exit code.
    """
    file_env = await _read_env(args.env_file)
    llm, bench = resolve_settings(file_env)
    if args.judge:
        bench = bench.model_copy(update={"judge_model": args.judge})
    cases = select_cases(args.case)
    routes = build_routes([m.strip() for m in args.models.split(",")], llm)
    judge_setup = None if args.no_judge else build_judge_model(bench, llm)
    if judge_setup is None:
        _say("No judge (no flag or key): code checks only.")
    ledger = JudgeLedger()
    summaries: list[Summary] = []
    results: list[ExampleResult] = []
    counts: dict[str, int] = {}
    for case in cases:
        examples = examples_of(case, args.golden_dir, args.limit)
        counts[case.name] = len(examples)
        target = _target(case, args.golden_dir, judge_setup, bench)
        for route in routes:
            done = await run_route(
                target, route, examples, ledger, args.concurrency or bench.concurrency
            )
            results.extend(done)
            summaries.append(summarize(case.name, route.name, done))
            _say(_line(summaries[-1]))
    if not args.no_report:
        info = _run_info(cases, routes, counts, judge_setup, ledger)
        _write_report(args.report, summaries, results, info)
    return 0


def _line(summary: Summary) -> str:
    if summary.skipped:
        return f"{summary.case:24} {summary.model:13} skipped"
    accuracy = NO_DATA if summary.accuracy is None else f"{summary.accuracy:.2f}"
    p50 = NO_DATA if summary.p50_seconds is None else f"{summary.p50_seconds:.1f}s"
    return (
        f"{summary.case:24} {summary.model:13} accuracy {accuracy} p50 {p50} "
        f"answered {summary.answered}/{summary.total}"
    )


def _judge_cost(setup: JudgeSetup | None, ledger: JudgeLedger) -> float | None:
    price = PRICES.get(setup.model.model_name) if setup else None
    if setup is None or price is None:
        return None
    return (
        ledger.input_tokens * price[0] + ledger.output_tokens * price[1]
    ) / TOKENS_PER_MILLION


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Arguments; the process arguments when ``None``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    os.environ.setdefault(QUIET_ENV, "1")
    return asyncio.run(run(args))
