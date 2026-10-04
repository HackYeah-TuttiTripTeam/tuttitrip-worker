"""Places domain: OSM candidates from recorded answers (never the network)."""

import asyncio
import json
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import httpx
import pytest
from dbos import DBOSClient, PortableWorkflowError
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql.expression import ClauseElement

from tests.helpers import enqueue
from tuttitrip_worker.contracts import (
    CONTRACT_VERSION,
    ErrorCode,
    FetchPlaceCandidatesOutput,
    Workflow,
)
from tuttitrip_worker.places import steps, workflows
from tuttitrip_worker.places.logic.city import (
    EURO_COUNTRIES,
    OTHER_CURRENCIES,
    parse_city,
)
from tuttitrip_worker.places.logic.hours import monday_of, weekly_hours
from tuttitrip_worker.places.logic.mapping import to_place_row
from tuttitrip_worker.places.logic.slug import slugify
from tuttitrip_worker.places.logic.taxonomy import RULES, classify, overpass_query
from tuttitrip_worker.places.schemas import GeocodedCity, PlaceRow, RefreshState
from tuttitrip_worker.shared.config.settings import Settings, get_settings
from tuttitrip_worker.shared.db.tables import places

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "osm"
NOMINATIM = json.loads((FIXTURES / "nominatim_sopot.json").read_text("utf-8"))
OVERPASS = json.loads((FIXTURES / "overpass_sopot.json").read_text("utf-8"))
MONDAY = date(2026, 9, 28)
SOPOT = 1553144

# The backend's catalog vocabulary (tuttitrip.places.schemas); the database
# CHECK constraints reject anything else.
CATEGORIES = {
    "attraction", "museum", "restaurant", "cafe", "park", "viewpoint",
    "shopping", "nightlife", "entertainment", "lodging",
}  # fmt: skip
INTERESTS = {
    "history", "architecture", "art", "museums", "science", "religion", "music",
    "nature", "parks", "views", "beaches", "sport", "adventure", "family", "kids",
    "nightlife", "shopping", "markets", "local_food", "street_food", "relaxation",
    "animals", "playground", "water", "cycling", "wellness",
}  # fmt: skip
CUISINES = {
    "polish", "italian", "french", "german", "british", "spanish", "greek",
    "turkish", "middle_eastern", "indian", "chinese", "japanese", "thai",
    "vietnamese", "mexican", "american", "international",
}  # fmt: skip
DIETS = {
    "vegetarian", "vegan", "pescatarian", "gluten_free", "lactose_free",
    "nut_free", "halal", "kosher",
}  # fmt: skip
AMENITIES = {
    "pool", "kitchen", "parking", "family_room", "wifi", "air_conditioning",
    "breakfast", "pets_allowed", "elevator", "wheelchair_accessible",
    "washing_machine", "balcony", "crib", "playground",
}  # fmt: skip


def element(**tags: str) -> dict[str, Any]:
    return {"type": "node", "id": 1, "lat": 54.44, "lon": 18.56, "tags": tags}


# --- pure logic ---------------------------------------------------------------------


def test_slug_follows_the_contract_rule() -> None:
    assert slugify("Gdańsk, Polska") == "gdansk-polska"
    assert slugify("  Łódź ") == "lodz"
    assert slugify("Kraków") == "krakow"
    assert not slugify("???")


def test_the_query_is_one_area_query_with_every_rule_key() -> None:
    query = overpass_query(SOPOT)
    assert "area(3601553144)->.city;" in query
    assert query.rstrip().endswith("out center tags;")
    assert query.count("(area.city);") == len({(r.key, r.needs) for r in RULES})
    for rule in RULES:
        assert rule.value in query


def test_every_rule_speaks_the_catalog_vocabulary() -> None:
    for rule in RULES:
        assert rule.category in CATEGORIES
        assert set(rule.tags) <= INTERESTS


def test_recorded_elements_all_map_to_valid_rows() -> None:
    rows = [to_place_row(e, "Europe/Warsaw", MONDAY) for e in OVERPASS["elements"]]
    assert all(rows)
    for row in rows:
        assert row is not None
        assert row.category in CATEGORIES
        assert set(row.tags) <= INTERESTS
        assert row.cuisine is None or row.cuisine in CUISINES
        assert set(row.diet_tags) <= DIETS
        assert set(row.amenities) <= AMENITIES
        assert (row.amenities == []) or row.category == "lodging"
        assert 54 < row.lat < 55
        assert 18 < row.lon < 19
    assert {row.category for row in rows if row} >= {"restaurant", "lodging", "park"}
    assert any(row.opening_hours for row in rows if row)


def test_a_restaurant_gets_cuisine_diet_wheelchair_and_hours() -> None:
    row = to_place_row(
        element(
            amenity="restaurant",
            name="Trattoria",
            cuisine="pizza;italian",
            **{"diet:vegan": "yes", "diet:halal": "no"},
            wheelchair="yes",
            opening_hours="Mo-Fr 12:00-22:00",
        ),
        "Europe/Warsaw",
        MONDAY,
    )
    assert row is not None
    assert (row.category, row.cuisine, row.diet_tags) == (
        "restaurant",
        "italian",
        ["vegan"],
    )
    assert row.wheelchair is True
    assert row.opening_hours == {
        "weekly": {
            day: [{"open": "12:00", "close": "22:00"}]
            for day in ("mon", "tue", "wed", "thu", "fri")
        },
        "closed_dates": [],
    }


def test_a_hotel_gets_amenities_and_a_way_uses_its_center() -> None:
    way = {
        "type": "way",
        "id": 7,
        "center": {"lat": 54.4, "lon": 18.5},
        "tags": {
            "tourism": "hotel",
            "name": "Grand",
            "internet_access": "wlan",
            "swimming_pool": "yes",
            "dog": "no",
        },
    }
    row = to_place_row(way, "Europe/Warsaw", MONDAY)
    assert row is not None
    assert (row.osm_type, row.osm_id, row.lat, row.lon) == ("way", 7, 54.4, 18.5)
    assert row.amenities == ["wifi", "pool"]


def test_wheelchair_and_indoor_are_conservative() -> None:
    limited = to_place_row(
        element(tourism="museum", name="M", wheelchair="limited"),
        "Europe/Warsaw",
        MONDAY,
    )
    assert limited is not None
    assert (limited.wheelchair, limited.indoor) == (False, True)
    unknown = to_place_row(
        element(amenity="restaurant", name="R"), "Europe/Warsaw", MONDAY
    )
    assert unknown is not None
    assert (unknown.wheelchair, unknown.indoor) == (None, None)
    explicit = classify({"leisure": "park", "indoor": "yes"})
    assert explicit is not None
    assert explicit.indoor is True
    park = classify({"leisure": "park"})
    assert park is not None
    assert park.indoor is False


@pytest.mark.parametrize(
    "bad",
    [
        {"tourism": "museum"},  # no name
        {"name": "x", "shop": "bakery"},  # no rule
        {"name": "x", "amenity": "place_of_worship"},  # not notable (no wikidata)
    ],
    ids=["unnamed", "no-rule", "plain-church"],
)
def test_elements_without_a_name_or_rule_are_dropped(bad: dict[str, str]) -> None:
    assert to_place_row(element(**bad), "Europe/Warsaw", MONDAY) is None


def test_a_notable_church_is_kept() -> None:
    row = to_place_row(
        element(amenity="place_of_worship", name="Kościół", wikidata="Q1"),
        "Europe/Warsaw",
        MONDAY,
    )
    assert row is not None
    assert row.tags == ["religion", "architecture"]


def test_hours_split_at_midnight_and_close_at_24() -> None:
    hours = weekly_hours("Fr 20:00-02:00", "Europe/Warsaw", MONDAY)
    assert hours is not None
    assert hours["weekly"]["fri"] == [{"open": "20:00", "close": "24:00"}]
    assert hours["weekly"]["sat"][0] == {"open": "00:00", "close": "02:00"}
    always = weekly_hours("24/7", "Europe/Warsaw", MONDAY)
    assert always is not None
    assert always["weekly"]["sun"] == [{"open": "00:00", "close": "24:00"}]


@pytest.mark.parametrize("expression", ["garbage x", "off", "", "x" * 600])
def test_unparseable_or_never_open_hours_mean_no_hours(expression: str) -> None:
    assert weekly_hours(expression, "Europe/Warsaw", MONDAY) is None


def test_the_reference_week_starts_on_the_citys_monday() -> None:
    sunday_night_utc = datetime(2026, 10, 4, 23, 30, tzinfo=UTC)  # Monday in Warsaw
    assert monday_of(sunday_night_utc, "Europe/Warsaw") == date(2026, 10, 5)
    assert monday_of(sunday_night_utc, "UTC") == date(2026, 9, 28)


def test_nominatim_answer_becomes_a_city() -> None:
    city = parse_city(NOMINATIM)
    assert city is not None
    assert (city.relation_id, city.name, city.country) == (SOPOT, "Sopot", "PL")
    assert (city.timezone, city.currency) == ("Europe/Warsaw", "PLN")
    assert city.bbox_south < city.center_lat < city.bbox_north
    assert city.bbox_west < city.center_lon < city.bbox_east
    assert parse_city([]) is None
    assert parse_city([{**NOMINATIM[0], "osm_type": "node"}]) is None


# --- statements ---------------------------------------------------------------------


def sql(statement: ClauseElement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def row(osm_id: int = 1) -> PlaceRow:
    mapped = to_place_row(
        {**element(amenity="cafe", name="Kawa"), "id": osm_id}, "UTC", MONDAY
    )
    assert mapped is not None
    return mapped


def test_places_are_upserted_by_osm_key_and_sheet_rows_are_left_alone() -> None:
    text = sql(steps.build_places_upsert("sopot", [row(), row(2)]))
    assert "ON CONFLICT (osm_type, osm_id) DO UPDATE" in text
    assert "WHERE places.source = " in text
    compiled = steps.build_places_upsert("sopot", [row()]).compile(
        dialect=postgresql.dialect()
    )
    assert "osm" in compiled.params.values()
    assert "CASE WHEN places.hours_verified THEN places.opening_hours" in text
    assert "city_slug = " not in text.split("DO UPDATE")[1]
    values = steps.place_values("sopot", row())
    assert (values["source"], values["hours_verified"]) == ("osm", False)


def test_unknown_hours_are_sql_null_not_json_null() -> None:
    hours_type = places.c.opening_hours.type
    assert isinstance(hours_type, JSONB)
    assert hours_type.none_as_null is True


def test_an_existing_city_is_never_overwritten() -> None:
    city = parse_city(NOMINATIM)
    assert city is not None
    text = sql(steps.build_city_insert("sopot", city))
    assert "ON CONFLICT (slug) DO NOTHING" in text
    assert "relation_id" not in text


def test_state_queries_read_city_marker_and_attempts() -> None:
    assert "FROM cities" in sql(steps.build_city_select("sopot"))
    marker = sql(steps.build_marker_select("sopot"))
    assert "job_results.workflow_id" in marker
    recent = sql(steps.build_recent_attempts_select(datetime.now(UTC)))
    assert "count(*)" in recent
    assert "job_results.result ->>" in recent
    assert "LIKE" in recent


# --- the steps over a fake network ---------------------------------------------------


class Net:
    """Fake Nominatim and Overpass; keeps every request."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []
        self.overpass_answers: list[httpx.Response] = []
        self.nominatim: list[dict[str, Any]] = NOMINATIM

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.host == "nominatim.openstreetmap.org":
            return httpx.Response(200, json=self.nominatim)
        if self.overpass_answers:
            return self.overpass_answers.pop(0)
        return httpx.Response(200, json=OVERPASS)

    def to(self, host: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == host]


class Db:
    """Fake database: the statements the real steps executed."""

    def __init__(self) -> None:
        self.executed: list[ClauseElement] = []
        self.cities: list[tuple[str, dict[str, Any]]] = []
        self.markers: list[tuple[str, int, int]] = []
        self.reservations: list[str] = []
        self.budget_left = True


class FakeConnection:
    def __init__(self, db: Db) -> None:
        self.db = db

    async def execute(self, statement: ClauseElement) -> None:
        self.db.executed.append(statement)


@pytest.fixture
def net(monkeypatch: pytest.MonkeyPatch) -> Net:
    fake = Net()

    async def record_sleep(seconds: float) -> None:
        fake.sleeps.append(seconds)

    monkeypatch.setattr(steps, "_sleep", record_sleep)
    monkeypatch.setattr(
        steps.httpx,
        "AsyncClient",
        partial(httpx.AsyncClient, transport=httpx.MockTransport(fake.handle)),
    )
    return fake


@pytest.fixture
def db(monkeypatch: pytest.MonkeyPatch) -> Db:
    fake = Db()

    @asynccontextmanager
    async def fake_transaction() -> AsyncGenerator[FakeConnection]:
        yield FakeConnection(fake)

    async def fake_store_city(slug: str, city: dict[str, Any]) -> None:
        fake.cities.append((slug, city))

    async def fake_mark(slug: str, relation_id: int, stored: int) -> None:
        fake.markers.append((slug, relation_id, stored))

    async def fake_reserve(workflow_id: str, slug: str) -> bool:
        del slug
        fake.reservations.append(workflow_id)
        return fake.budget_left

    monkeypatch.setattr(steps, "reserve_overpass_slot", fake_reserve)
    monkeypatch.setattr(steps, "transaction", fake_transaction)
    monkeypatch.setattr(steps, "store_city", fake_store_city)
    monkeypatch.setattr(steps, "mark_fetched", fake_mark)
    monkeypatch.setattr(workflows, "PAUSE_BETWEEN_QUERIES_SEC", 0.01)
    return fake


def known(state: RefreshState) -> Callable[[str], Any]:
    async def load(slug: str) -> dict[str, Any]:
        del slug
        return state.model_dump(mode="json")

    return load


def run(
    client: DBOSClient, dbos: Settings, body: dict[str, Any]
) -> FetchPlaceCandidatesOutput:
    payload = {"contract_version": CONTRACT_VERSION, **body}
    handle = enqueue(client, dbos, Workflow.FETCH_PLACE_CANDIDATES, payload)
    return FetchPlaceCandidatesOutput.model_validate(handle.get_result())


def fail(
    client: DBOSClient, dbos: Settings, body: dict[str, Any]
) -> PortableWorkflowError:
    payload = {"contract_version": CONTRACT_VERSION, **body}
    handle = enqueue(client, dbos, Workflow.FETCH_PLACE_CANDIDATES, payload)
    with pytest.raises(PortableWorkflowError) as info:
        handle.get_result()
    return info.value


def test_a_new_city_is_geocoded_queried_once_and_stored(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))

    output = run(client, dbos, {"city_query": "Sopot"})

    assert (output.city_slug, output.source, output.refreshed) == ("sopot", "osm", True)
    assert output.stored == len(OVERPASS["elements"])
    assert len(net.to("nominatim.openstreetmap.org")) == 1
    assert len(net.to("overpass-api.de")) == 1  # one query for the whole city
    assert [slug for slug, _ in db.cities] == ["sopot"]
    assert db.cities[0][1]["relation_id"] == SOPOT
    assert db.markers == [("sopot", SOPOT, output.stored)]
    upserts = [sql(s) for s in db.executed]
    assert upserts
    assert all("INSERT INTO places" in text for text in upserts)


def test_the_overpass_request_is_a_single_area_query(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    run(client, dbos, {"city_query": "Sopot"})
    del db
    (request,) = net.to("overpass-api.de")
    assert request.method == "POST"
    (query,) = parse_qs(request.content.decode())["data"]
    assert "area(3601553144)->.city;" in query
    assert query.endswith("out center tags;")


def test_every_osm_request_names_tuttitrip(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    run(client, dbos, {"city_query": "Sopot"})
    del db
    assert len(net.requests) == 2
    for request in net.requests:
        agent = request.headers["User-Agent"]
        assert agent.startswith("TuttiTrip/")
        assert "github.com/HackYeah-TuttiTripTeam" in agent  # contact


def test_a_recently_fetched_city_asks_nobody(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState(fresh=True)))

    output = run(client, dbos, {"city_slug": "sopot"})

    assert (output.refreshed, output.stored) == (False, 0)
    assert net.requests == []
    assert (db.executed, db.cities, db.markers) == ([], [], [])


def test_a_known_relation_skips_nominatim(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RefreshState(relation_id=SOPOT, timezone="Europe/Warsaw")
    monkeypatch.setattr(steps, "load_refresh_state", known(state))

    output = run(client, dbos, {"city_slug": "sopot"})

    assert output.refreshed is True
    assert net.to("nominatim.openstreetmap.org") == []
    assert len(net.to("overpass-api.de")) == 1
    assert db.cities == []  # the city row already exists


def test_a_sheet_city_is_geocoded_by_its_name_and_country(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RefreshState(city_name="Sopot", country="PL", timezone="Europe/Warsaw")
    monkeypatch.setattr(steps, "load_refresh_state", known(state))

    run(client, dbos, {"city_slug": "sopot"})
    del db

    (request,) = net.to("nominatim.openstreetmap.org")
    assert request.url.params["q"] == "Sopot"
    assert request.url.params["countrycodes"] == "pl"


def test_429_makes_the_step_wait_at_least_30_seconds_and_retry(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    net.overpass_answers = [httpx.Response(429), httpx.Response(504)]
    pauses: list[float] = []
    real_sleep = asyncio.sleep

    async def fast_sleep(seconds: float) -> None:
        pauses.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    output = run(client, dbos, {"city_query": "Sopot"})
    del db

    assert output.stored > 0
    assert len(net.to("overpass-api.de")) == 3  # 429, 504, then the answer
    retry_pauses = [p for p in pauses if p >= 30]
    assert retry_pauses[:2] == [30.0, 60.0]  # at least 30 s, then backing off


def test_overpass_never_runs_two_queries_at_once() -> None:
    async def scenario() -> int:
        running = peak = 0

        async def query() -> None:
            nonlocal running, peak
            async with steps._gate:
                running += 1
                peak = max(peak, running)
                await asyncio.sleep(0)
                running -= 1

        await asyncio.gather(*(query() for _ in range(5)))
        return peak

    assert asyncio.run(scenario()) == 1


def test_the_pauses_follow_the_policies() -> None:
    assert steps.RETRY_PAUSE_SEC >= 30
    assert steps.REQUEST_SPACING_SEC >= 1  # Nominatim: 1 request a second
    assert workflows.PAUSE_BETWEEN_QUERIES_SEC >= 1


def test_the_gate_stays_closed_for_a_second_after_every_request(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    run(client, dbos, {"city_query": "Sopot"})
    del db
    assert len(net.requests) == 2  # Nominatim, then Overpass
    assert net.sleeps == [1.0, 1.0]  # one second inside the gate after each


def test_retry_after_longer_than_30_s_is_waited_out(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    net.overpass_answers = [httpx.Response(429, headers={"Retry-After": "120"})]
    real_sleep = asyncio.sleep

    async def fast_sleep(seconds: float) -> None:
        del seconds
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    run(client, dbos, {"city_query": "Sopot"})
    del db
    # 90 s here (still holding the gate) plus the 30 s DBOS retry pause.
    assert 90.0 in net.sleeps


def test_an_answer_without_elements_is_a_clear_error(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    net.overpass_answers = [httpx.Response(200, json={"version": 0.6})]
    error = fail(client, dbos, {"city_query": "Sopot"})
    del db
    assert "elements" in str(error)


def test_the_same_slug_for_another_osm_city_is_a_conflict(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = RefreshState(relation_id=42, timezone="Europe/London")
    monkeypatch.setattr(steps, "load_refresh_state", known(stored))

    error = fail(client, dbos, {"city_query": "Sopot"})

    assert error.code == ErrorCode.SLUG_CONFLICT.value
    assert net.to("overpass-api.de") == []
    assert db.reservations == []
    assert db.cities == []


def test_the_same_city_again_after_the_refresh_period_is_fine(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = RefreshState(relation_id=SOPOT, timezone="Europe/Warsaw")
    monkeypatch.setattr(steps, "load_refresh_state", known(stored))
    assert run(client, dbos, {"city_query": "Sopot"}).refreshed is True
    del net, db


def test_a_runtime_error_remark_is_retried(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    busy = {"elements": [], "remark": "runtime error: Query timed out"}
    net.overpass_answers = [httpx.Response(200, json=busy)]
    real_sleep = asyncio.sleep

    async def fast_sleep(seconds: float) -> None:
        await real_sleep(0)
        del seconds

    monkeypatch.setattr(asyncio, "sleep", fast_sleep)
    assert run(client, dbos, {"city_query": "Sopot"}).stored > 0
    del db


def test_duplicate_elements_are_one_row_so_a_repeat_adds_nothing(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    doubled = {"elements": OVERPASS["elements"] * 2}
    net.overpass_answers = [httpx.Response(200, json=doubled)] * 2

    first = run(client, dbos, {"city_query": "Sopot"})
    second = run(client, dbos, {"city_query": "Sopot"})
    del db

    assert first.stored == second.stored == len(OVERPASS["elements"])


def test_big_cities_are_written_in_batches(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    monkeypatch.setattr(steps, "BATCH_SIZE", 10)
    del net
    run(client, dbos, {"city_query": "Sopot"})
    assert len(db.executed) == -(-len(OVERPASS["elements"]) // 10)


def test_a_spent_daily_budget_stops_before_any_request(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    db.budget_left = False

    error = fail(client, dbos, {"city_query": "Sopot"})

    assert error.code == ErrorCode.RATE_LIMITED.value
    assert net.to("overpass-api.de") == []  # reserved before any Overpass call
    assert len(db.reservations) == 1


class Ledger:
    """In-memory stand-in for the reservation rows in job_results."""

    def __init__(self) -> None:
        self.rows: list[str] = []
        self.locked = False


class LedgerConnection:
    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger

    async def execute(self, statement: ClauseElement) -> LedgerResult:
        text = sql(statement)
        if "pg_advisory_xact_lock" in text:
            self.ledger.locked = True
            return LedgerResult(0)
        assert self.ledger.locked, "the lock comes before reading or writing"
        if text.startswith("INSERT"):
            compiled = statement.compile(dialect=postgresql.dialect())
            self.ledger.rows.append(str(compiled.params["workflow_id"]))
            return LedgerResult(0)
        if "job_results.workflow_id =" in text:  # this workflow's own row
            params = statement.compile(dialect=postgresql.dialect()).params
            return LedgerResult(self.ledger.rows.count(params["workflow_id_1"]))
        return LedgerResult(len(self.ledger.rows))  # the last 24 hours


class LedgerResult:
    def __init__(self, value: int) -> None:
        self.value = value

    def scalar_one(self) -> int:
        return self.value


def test_failed_attempts_count_and_a_recovered_workflow_reserves_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = Ledger()

    @asynccontextmanager
    async def fake_transaction() -> AsyncGenerator[LedgerConnection]:
        ledger.locked = False
        yield LedgerConnection(ledger)

    monkeypatch.setattr(steps, "transaction", fake_transaction)
    monkeypatch.setenv("TUTTITRIP_OSM__OVERPASS_DAILY_LIMIT", "2")
    get_settings.cache_clear()

    async def scenario() -> list[bool]:
        # wf-a reserves and then fails (nothing is marked fetched): it still counts.
        return [
            await steps.reserve_overpass_slot("wf-a", "sopot"),
            await steps.reserve_overpass_slot("wf-b", "gdansk"),
            await steps.reserve_overpass_slot("wf-c", "lodz"),  # over the limit
            await steps.reserve_overpass_slot("wf-a", "sopot"),  # recovery
        ]

    assert asyncio.run(scenario()) == [True, True, False, True]
    assert ledger.rows == ["osm-attempt:wf-a", "osm-attempt:wf-b"]  # no 3rd, no repeat


def test_an_unknown_city_is_reported(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    net.nominatim = []

    error = fail(client, dbos, {"city_query": "Nibylandia"})

    assert error.code == ErrorCode.CITY_NOT_FOUND.value
    assert net.to("overpass-api.de") == []
    assert db.cities == []


def test_a_slug_nobody_fetched_needs_a_query(
    client: DBOSClient,
    dbos: Settings,
    net: Net,
    db: Db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(steps, "load_refresh_state", known(RefreshState()))
    error = fail(client, dbos, {"city_slug": "nibylandia"})
    assert error.code == ErrorCode.CITY_NOT_FOUND.value
    assert net.requests == []
    del db


def test_currencies_come_from_a_static_table() -> None:
    assert "DE" in EURO_COUNTRIES
    assert OTHER_CURRENCIES["PL"] == "PLN"
    assert not EURO_COUNTRIES & OTHER_CURRENCIES.keys()


def test_geocoded_city_round_trips_through_json() -> None:
    city = parse_city(NOMINATIM)
    assert city is not None
    assert GeocodedCity.model_validate(city.model_dump(mode="json")) == city


def test_the_refresh_marker_is_fresh_for_refresh_days_only() -> None:
    assert steps.marker_id("sopot") == "osm-fetch:sopot"
    assert timedelta(days=Settings(_env_file=None).osm.refresh_days) >= timedelta(
        days=1
    )
