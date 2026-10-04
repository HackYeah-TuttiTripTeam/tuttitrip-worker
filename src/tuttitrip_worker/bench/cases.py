"""The benchmark cases: how to run a worker use case on a golden example.

Each case drives the production function of its use case (the same prompt,
output validator and retries as in the worker) with the model under test
swapped in by ``Subject.using``. Cases of pull requests that are not merged
yet are imported when they run and report :class:`CaseUnavailableError` until
they land.
"""

import asyncio
import importlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tuttitrip_worker.accommodation import agents as offer_agents
from tuttitrip_worker.accommodation.services.assess import assess_offer
from tuttitrip_worker.bench.agents import Subject
from tuttitrip_worker.bench.constants import SEED_PREFIX, TRIP_ID, Role
from tuttitrip_worker.bench.errors import (
    CaseUnavailableError,
    InvalidOutputError,
    ProviderError,
)
from tuttitrip_worker.bench.logic import scoring
from tuttitrip_worker.bench.logic.derive import verdict_examples
from tuttitrip_worker.bench.schemas import Example, Json, Scored
from tuttitrip_worker.contracts import RequirementLabel
from tuttitrip_worker.linter import agents as plan_agents
from tuttitrip_worker.linter.logic.quotes import split_items
from tuttitrip_worker.planning.agents import planner_agent

type RunFn = Callable[[Example, Subject, Path], Awaitable[Json]]
type ScoreFn = Callable[[Example, Json], Scored]
type DeriveFn = Callable[[list[Example]], list[Example]]

EXPENSES_AGENTS = "tuttitrip_worker.expenses.agents"
"""Module of ``parse_expense_text`` and ``read_receipt`` (worker PR #45)."""

MATCH_AGENTS = "tuttitrip_worker.linter.agents"
"""Module of the place matcher (worker PR #47)."""

MATCH_CANDIDATES = "tuttitrip_worker.linter.logic.candidates"
"""Candidate ranking of the place matcher (worker PR #47)."""

MATCH_SCHEMAS = "tuttitrip_worker.linter.schemas"
"""``CatalogPlace`` of the place matcher (worker PR #47)."""

MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".webp": "image/webp"}
"""Media type by image file suffix."""


@dataclass(frozen=True)
class Case:
    """One benchmark case."""

    name: str
    summary: str
    role: Role
    run: RunFn
    score: ScoreFn
    source: str | None = None
    """Golden directory when it differs from ``name`` (a derived case)."""
    derive: DeriveFn | None = None


def _optional(module: str, attribute: str | None = None) -> Any:  # ruff: ignore[any-type]
    """Import a module of an unmerged pull request.

    Args:
        module: Dotted module path.
        attribute: Name to take from it, or the module itself.

    Returns:
        The module or its attribute.

    Raises:
        CaseUnavailableError: The module or the name does not exist yet.
    """
    try:
        loaded = importlib.import_module(module)
        return loaded if attribute is None else getattr(loaded, attribute)
    except (ImportError, AttributeError) as error:
        msg = f"{module} is not in this build of the worker: {error}"
        raise CaseUnavailableError(msg) from error


async def run_offer(example: Example, subject: Subject, _golden: Path) -> Json:
    """``extract_offer_evidence``: quotes and verdicts of a pasted offer.

    Args:
        example: Input ``offer`` and ``requirements`` (``key``, ``label``).
        subject: Model under test; it plays the extractor and the judge.
        _golden: Golden directory (unused).

    Returns:
        ``{"evidence": {key: {"quotes": [{"text", "verdict"}]}}}``.
    """
    requirements: list[Json] = example.input["requirements"]
    keys = [item["key"] for item in requirements]
    labels = [
        RequirementLabel(key=item["key"], label=item["label"])
        for item in requirements
        if item.get("label")
    ]
    with subject.using(offer_agents.quote_extractor, offer_agents.requirement_judge):
        evidence = await assess_offer(example.input["offer"], keys, labels)
    return {
        "evidence": {
            item.requirement_key: {
                "quotes": [
                    {"text": quote.text, "verdict": quote.verdict}
                    for quote in item.quotes
                ]
            }
            for item in evidence
        }
    }


async def run_verdict(example: Example, subject: Subject, _golden: Path) -> Json:
    """Judge stage of the offer: what does each quote say about a requirement?

    Args:
        example: Input ``key``, ``label`` and ``quotes``.
        subject: Model under test (a decision model or a language model).
        _golden: Golden directory (unused).

    Returns:
        ``{"verdicts": [...]}``.

    Raises:
        InvalidOutputError: The model answered but not in the required shape.
        ProviderError: The model did not answer at all.
    """
    with subject.using(offer_agents.requirement_judge):
        judged = await offer_agents.judge_quotes(
            example.input["key"], example.input["label"], example.input["quotes"]
        )
    if judged is None:
        # The worker swallows both failures; the meter tells them apart.
        if subject.model.requests:
            msg = "the judge never produced a valid verdict"
            raise InvalidOutputError(msg)
        msg = "the judge model did not answer"
        raise ProviderError(msg)
    return {"verdicts": [verdict for verdict, _confidence in judged]}


async def run_plan(example: Example, subject: Subject, _golden: Path) -> Json:
    """``parse_pasted_plan``: items of a pasted plan, checked like the worker does.

    Args:
        example: Input ``text`` and ``city_slug``.
        subject: Model under test.
        _golden: Golden directory (unused).

    Returns:
        ``{"items": [...], "unread": [...]}`` (contract items and unread lines).
    """
    text = example.input["text"]
    with subject.using(plan_agents.parser_agent):
        draft = await plan_agents.read_pasted_plan(
            text, f"{SEED_PREFIX}-{example.id}", example.input["city_slug"]
        )
    items, unread = split_items(draft.items, text)
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "unread": [item.model_dump(mode="json") for item in unread],
    }


async def run_trip(example: Example, subject: Subject, _golden: Path) -> Json:
    """``generate_trip_plan``: a short plan draft for a request.

    Args:
        example: Input ``request``.
        subject: Model under test.
        _golden: Golden directory (unused).

    Returns:
        ``{"destination", "days", "highlights"}``.
    """
    with subject.using(planner_agent):
        result = await planner_agent.run(f"Trip {TRIP_ID}: {example.input['request']}")
    return result.output.model_dump(mode="json")


async def run_expense(example: Example, subject: Subject, _golden: Path) -> Json:
    """``parse_expense_text``: one typed sentence about an expense.

    Args:
        example: Input ``text`` and ``locale``.
        subject: Model under test.
        _golden: Golden directory (unused).

    Returns:
        The reading as the worker returns it.
    """
    agents = _optional(EXPENSES_AGENTS)
    with subject.using(agents.text_agent):
        reading = await agents.read_typed_expense(
            example.input["text"],
            f"{SEED_PREFIX}-{example.id}",
            example.input["locale"],
        )
    return reading.model_dump(mode="json")


async def run_receipt(example: Example, subject: Subject, golden: Path) -> Json:
    """``read_receipt``: a receipt or bank screenshot (an image of the set).

    Args:
        example: Input ``image``, a path relative to the case directory.
        subject: Model under test (it must accept images).
        golden: Directory of the case.

    Returns:
        The reading as the worker returns it.
    """
    agents = _optional(EXPENSES_AGENTS)
    path = golden / example.input["image"]
    data = await asyncio.to_thread(path.read_bytes)
    with subject.using(agents.receipt_reader):
        reading = await agents.read_receipt_image(data, MEDIA_TYPES[path.suffix])
    return reading.model_dump(mode="json")


async def run_match(example: Example, subject: Subject, _golden: Path) -> Json:
    """Place matching: which catalog place does a pasted item mean?

    The candidates are ranked by the worker's pure code; the model under test
    only picks one of them, or none.

    Args:
        example: Input ``name``, ``quote`` and the city's ``places``.
        subject: Model under test.
        _golden: Golden directory (unused).

    Returns:
        ``{"place_id": id or None, "candidates": [ids]}``.

    Raises:
        InvalidOutputError: The model answered but not in the required shape.
        ProviderError: The model did not answer at all.
    """
    agents = _optional(MATCH_AGENTS)
    rank = _optional(MATCH_CANDIDATES, "rank")
    catalog_place = _optional(MATCH_SCHEMAS, "CatalogPlace")
    places = [catalog_place(**place) for place in example.input["places"]]
    ranked = rank(example.input["name"], places)
    candidates = [place.place_id for place in ranked]
    if not ranked:
        return {"place_id": None, "candidates": candidates}
    with subject.using(agents.place_matcher):
        decided = await agents.choose_place(
            example.input["name"],
            example.input["quote"],
            f"{SEED_PREFIX}-{example.id}",
            ranked,
        )
    if decided is None:
        if subject.model.requests:
            msg = "the matcher never produced a valid pick"
            raise InvalidOutputError(msg)
        msg = "the matcher model did not answer"
        raise ProviderError(msg)
    pick, _confidence = decided
    return {"place_id": pick.place_id if pick else None, "candidates": candidates}


CASES: dict[str, Case] = {
    case.name: case
    for case in (
        Case(
            "extract_offer_evidence",
            "Quotes and verdicts for requirements from a pasted accommodation offer",
            Role.GENERATIVE,
            run_offer,
            scoring.score_offer,
        ),
        Case(
            "offer_verdict",
            "Judge stage of the offer: verdict of each quote (derived from the above)",
            Role.ANY,
            run_verdict,
            scoring.score_verdict,
            source="extract_offer_evidence",
            derive=verdict_examples,
        ),
        Case(
            "parse_pasted_plan",
            "Items of a plan pasted from a chatbot",
            Role.GENERATIVE,
            run_plan,
            scoring.score_plan,
        ),
        Case(
            "generate_trip_plan",
            "Short trip plan draft from a request",
            Role.GENERATIVE,
            run_trip,
            scoring.score_trip,
        ),
        Case(
            "parse_expense_text",
            "A typed sentence about an expense",
            Role.GENERATIVE,
            run_expense,
            scoring.score_expense,
        ),
        Case(
            "read_receipt",
            "A photo of a receipt or a bank screenshot",
            Role.GENERATIVE,
            run_receipt,
            scoring.score_receipt,
        ),
        Case(
            "match_places",
            "Which catalog place does a pasted item mean",
            Role.ANY,
            run_match,
            scoring.score_match,
        ),
    )
}
"""All cases by name."""
