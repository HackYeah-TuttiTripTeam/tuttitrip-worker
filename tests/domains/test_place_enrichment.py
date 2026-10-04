"""Web research of OSM places: source checks, SQL, the agent and the cost limit.

No network: the model is a ``FunctionModel`` that claims native web search.
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.profiles import ModelProfile
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql.expression import ClauseElement

from tuttitrip_worker.places import agents, steps, workflows
from tuttitrip_worker.places.logic.enrichment import clean
from tuttitrip_worker.places.schemas import (
    CleanFacts,
    CleanPrice,
    EnrichTarget,
    PlaceFacts,
    PriceFact,
)
from tuttitrip_worker.shared.config.settings import Settings, get_settings
from tuttitrip_worker.shared.llm.models import catalog

PAGE = "https://museum.example/visit"
TARGET = EnrichTarget(
    place_id="p1", name="Museum", category="museum", lat=1.0, lon=2.0, city="Lisboa"
)


def facts(**changes: object) -> PlaceFacts:
    base: dict[str, Any] = {
        "opening_hours": {"tue": ["09:30-17:30"], "wed": ["09:30-17:30"]},
        "hours_source_url": PAGE,
        "prices": [
            PriceFact(category="adult", amount=12, currency="eur", source_url=PAGE),
            PriceFact(category="child", amount=0, currency="EUR", source_url=PAGE),
        ],
        "visit_min": 90,
        "indoor": True,
        "child_friendly": True,
        "description": " A museum. ",
    }
    return PlaceFacts.model_validate(base | changes)


def test_sourced_facts_are_kept() -> None:
    result = clean(facts(), [PAGE + "/"])
    assert result.opening_hours == {
        "weekly": {
            "tue": [{"open": "09:30", "close": "17:30"}],
            "wed": [{"open": "09:30", "close": "17:30"}],
        },
        "closed_dates": [],
    }
    assert result.hours_source_url == PAGE
    assert [(p.category, p.amount, p.currency) for p in result.prices] == [
        ("adult", 12.0, "EUR"),
        ("child", 0.0, "EUR"),  # free entry is a sourced price of 0
    ]
    assert (result.visit_min, result.indoor, result.child_friendly) == (90, True, True)
    assert result.description == "A museum."


def test_a_source_the_run_never_saw_drops_prices_and_hours() -> None:
    result = clean(facts(), ["https://other.example/page"])
    assert result.opening_hours is None
    assert result.hours_source_url is None
    assert result.prices == []
    assert result.visit_min == 90  # an estimate needs no page


def test_no_source_url_no_price() -> None:
    priced = facts(
        prices=[PriceFact(category="adult", amount=5, currency="EUR", source_url="")]
    )
    assert clean(priced, [PAGE]).prices == []


@pytest.mark.parametrize(
    "hours",
    [{"tue": ["25:00-26:00"]}, {"tue": ["10:00-09:00"]}, {"tue": ["open all day"]}, {}],
)
def test_bad_hours_are_not_stored_even_in_part(hours: dict[str, list[str]]) -> None:
    result = clean(
        facts(opening_hours=hours | {"wed": ["09:00-10:00"]} if hours else hours),
        [PAGE],
    )
    assert result.opening_hours is None


def test_implausible_values_are_dropped() -> None:
    wild = facts(
        visit_min=5,
        prices=[PriceFact(category="vip", amount=5, currency="EUR", source_url=PAGE)],
    )
    result = clean(wild, [PAGE])
    assert result.visit_min is None
    assert result.prices == []


def test_a_failed_run_is_empty() -> None:
    assert clean(None, []) == CleanFacts()


def sql(statement: ClauseElement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_update_fills_gaps_and_never_replaces_known_hours() -> None:
    text = sql(
        steps.build_enrichment_update(
            "00000000-0000-0000-0000-000000000001",
            CleanFacts(
                opening_hours={"weekly": {}, "closed_dates": []}, hours_source_url=PAGE
            ),
            datetime.now(UTC),
        )
    )
    assert "coalesce(places.opening_hours" in text
    assert "hours_verified" not in text
    assert "places.source =" in text


def test_prices_are_upserted_unverified_and_a_verified_row_is_left_alone() -> None:
    price = CleanPrice(category="adult", amount=9, currency="EUR", source_url=PAGE)
    statement = steps.build_price_upsert(
        "00000000-0000-0000-0000-000000000001",
        CleanFacts(prices=[price]),
        datetime.now(UTC),
    )
    text = sql(statement)
    assert "ON CONFLICT (place_id, ticket_category) DO UPDATE" in text
    assert "place_prices.verified IS false" in text
    assert (
        statement.compile(dialect=postgresql.dialect()).params["verified_m0"] is False
    )


def test_targets_are_a_fixed_ordered_slice() -> None:
    text = sql(steps.build_targets_select("lizbona", ["attraction", "museum"], 20))
    assert "places.source =" in text
    assert "ORDER BY CASE places.category" in text
    assert "LIMIT" in text


def reply(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del messages
    body = facts().model_dump(mode="json")
    return ModelResponse(
        parts=[ToolCallPart(info.output_tools[0].name, json.dumps(body))],
        provider_details={
            "cost": 0.04,
            "annotations": [{"type": "url_citation", "url_citation": {"url": PAGE}}],
        },
    )


def search_capable() -> FunctionModel:
    profile = ModelProfile(supported_native_tools=frozenset({WebSearchTool}))
    return FunctionModel(reply, profile=profile)


def test_the_agent_returns_facts_cited_urls_and_cost() -> None:
    with catalog.override(search_capable()):
        result = asyncio.run(agents.research(TARGET, get_settings().enrich))
    assert result.facts == facts()
    assert result.cited_urls == [PAGE]
    assert result.cost_usd == pytest.approx(0.04)


def test_a_crashing_model_gives_no_facts_not_an_error() -> None:
    def boom(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        down = "provider down"
        raise RuntimeError(down)

    profile = ModelProfile(supported_native_tools=frozenset({WebSearchTool}))
    with catalog.override(FunctionModel(boom, profile=profile)):
        result = asyncio.run(agents.research(TARGET, get_settings().enrich))
    assert result.facts is None


def test_capabilities_follow_the_switches() -> None:
    base = len(agents.capabilities(Settings(_env_file=None).enrich))
    on = Settings(_env_file=None).enrich.model_copy(
        update={
            "web_fetch": True,
            "thinking": "low",
            "planning": True,
            "compaction": True,
        }
    )
    assert len(agents.capabilities(on)) == base + 4


def test_the_city_stops_at_the_cost_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TUTTITRIP_ENRICH__CONCURRENCY", "2")
    monkeypatch.setenv("TUTTITRIP_ENRICH__MAX_COST_USD", "0.05")
    get_settings.cache_clear()
    researched: list[str] = []
    stored: list[list[str]] = []

    async def targets(slug: str) -> list[dict[str, Any]]:
        del slug
        return [
            TARGET.model_copy(update={"place_id": f"p{i}"}).model_dump()
            for i in range(6)
        ]

    async def research_place(target: dict[str, Any]) -> dict[str, Any]:
        researched.append(target["place_id"])
        return {
            "place_id": target["place_id"],
            "facts": CleanFacts().model_dump(),
            "cost_usd": 0.03,
        }

    async def store(results: list[dict[str, Any]]) -> int:
        stored.append([r["place_id"] for r in results])
        return len(results)

    async def progress(stage: str, percent: int) -> None:
        del stage, percent

    monkeypatch.setattr(steps, "select_research_targets", targets)
    monkeypatch.setattr(steps, "research_place", research_place)
    monkeypatch.setattr(steps, "store_research", store)
    monkeypatch.setattr(workflows, "report_progress", progress)

    count = asyncio.run(workflows._enrich("lizbona"))

    assert researched == ["p0", "p1"]  # 0.06 >= 0.05 after the first chunk
    assert stored == [["p0", "p1"]]
    assert count == 2
    get_settings.cache_clear()
