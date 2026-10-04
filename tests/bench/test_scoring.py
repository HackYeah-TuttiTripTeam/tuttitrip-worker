"""Deterministic scoring of the benchmark cases."""

import pytest

from tuttitrip_worker.bench.logic import scoring
from tuttitrip_worker.bench.logic.compare import fold, name_set, text_covers
from tuttitrip_worker.bench.schemas import Example, Scored


def example(expected: dict[str, object], **input_: object) -> Example:
    return Example(id="pl-00", lang="pl", input=dict(input_), expected=expected)


def test_fold_removes_case_and_diacritics() -> None:
    assert fold("  Łódź  Żółć ") == "lodz zolc"
    assert not fold(None)


def test_text_covers_accepts_containment_and_empty_pairs() -> None:
    assert text_covers("Wawel", "Rynek") == pytest.approx(0.0)
    assert text_covers("Wawel", "Zamek na Wawelu") == pytest.approx(1.0)
    assert text_covers("Zamek na Wawelu", "zamek na wawelu, krakow") == pytest.approx(
        1.0
    )
    assert text_covers(None, "") == pytest.approx(1.0)
    assert text_covers("a", "") == pytest.approx(0.0)


def test_name_set_is_f1_over_folded_names() -> None:
    assert name_set([], []) == pytest.approx(1.0)
    assert name_set(["Ania"], ["ania"]) == pytest.approx(1.0)
    assert name_set(["Ania", "Bartek"], ["Ania"]) == pytest.approx(2 / 3)
    assert name_set(["Ania"], ["Ola"]) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("verdicts", "state"),
    [
        ([], "unconfirmed"),
        (["present"], "present"),
        (["absent", "absent"], "absent"),
        (["present", "absent"], "conflicting"),
        (["present", "not_applicable"], "present"),
        (["not_applicable"], "not_applicable"),
        ([None], "unjudged"),
    ],
)
def test_requirement_state(verdicts: list[str | None], state: str) -> None:
    assert scoring.requirement_state(verdicts) == state


OFFER = example(
    {
        "evidence": {
            "pool": {"quotes": [{"text": "kryty basen", "verdict": "present"}]},
            "sauna": {"quotes": []},
        }
    }
)


def test_offer_scores_state_and_quote_recall() -> None:
    answer = {
        "evidence": {
            "pool": {
                "quotes": [{"text": "Kryty basen z jacuzzi", "verdict": "present"}]
            },
            "sauna": {"quotes": []},
        }
    }
    scored = scoring.score_offer(OFFER, answer)
    assert scored.deterministic == pytest.approx(1.0)
    assert scored.judge is not None
    assert scored.judge.reference == {"pool": ["kryty basen"], "sauna": []}


def test_offer_penalizes_a_wrong_state_and_an_invented_quote() -> None:
    answer = {
        "evidence": {
            "pool": {"quotes": [{"text": "basen", "verdict": "absent"}]},
            "sauna": {"quotes": [{"text": "sauna", "verdict": "present"}]},
        }
    }
    scored = scoring.score_offer(OFFER, answer)
    assert scored.parts["pool:state"] == pytest.approx(0.0)
    assert scored.parts["sauna:state"] == pytest.approx(0.0)
    assert scored.deterministic < 1.0


def test_offer_survives_garbage_answers() -> None:
    assert scoring.score_offer(OFFER, {"evidence": "nope"}).deterministic < 1.0


def test_verdict_compares_position_by_position() -> None:
    sample = example({"verdicts": ["present", "absent"]})
    assert scoring.score_verdict(
        sample, {"verdicts": ["present", "absent"]}
    ).deterministic == pytest.approx(1.0)
    assert scoring.score_verdict(
        sample, {"verdicts": ["present"]}
    ).deterministic == pytest.approx(0.5)


PLAN = example(
    {
        "items": [
            {
                "place_name": "Wawel",
                "day": 1,
                "start_time": "09:00",
                "end_time": None,
                "amount_minor": 4000,
                "currency": "PLN",
                "transport": None,
                "address": "Wawel 5",
                "city": "Kraków",
            },
            {
                "place_name": "Sukiennice",
                "day": 1,
                "start_time": None,
                "end_time": None,
                "amount_minor": None,
                "currency": None,
                "transport": "walk",
                "address": None,
                "city": None,
            },
        ]
    }
)


def item(name: str, **fields: object) -> dict[str, object]:
    base: dict[str, object] = {
        "place_name": name,
        "day": None,
        "start_time": None,
        "end_time": None,
        "amount_minor": None,
        "currency": None,
        "transport": None,
        "address": None,
        "city": None,
    }
    return base | fields


def test_plan_pairs_items_by_name_and_checks_fields() -> None:
    answer = {
        "items": [
            item("Sukiennice", day=1, transport="walk"),
            item("Wawel", day=1, start_time="09:00", amount_minor=4000, currency="PLN"),
        ]
    }
    scored = scoring.score_plan(PLAN, answer)
    assert scored.parts == {"recall": 1.0, "precision": 1.0, "fields": 1.0}


def test_plan_penalizes_missing_extra_and_wrong_items() -> None:
    answer = {
        "items": [item("Wawel", day=2, amount_minor=4000, currency="PLN"), item("Nowy")]
    }
    scored = scoring.score_plan(PLAN, answer)
    assert scored.parts["recall"] == pytest.approx(0.5)
    assert scored.parts["precision"] == pytest.approx(0.5)
    assert scored.parts["fields"] < 1.0


def test_plan_without_items_expects_no_items() -> None:
    empty = example({"items": []})
    assert scoring.score_plan(empty, {"items": []}).deterministic == pytest.approx(1.0)
    assert scoring.score_plan(empty, {"items": [item("X")]}).parts[
        "precision"
    ] == pytest.approx(0.0)
    assert scoring.score_plan(empty, {"items": []}).judge is None


TRIP = example(
    {
        "destination": "Kraków|Krakow",
        "days": 3,
        "highlights": ["Wawel"],
        "criteria": "c",
    }
)


def test_trip_checks_destination_days_and_size() -> None:
    good = {"destination": "Krakow, Polska", "days": 3, "highlights": ["Wawel"]}
    assert scoring.score_trip(TRIP, good).deterministic == pytest.approx(1.0)
    bad = {"destination": "Gdańsk", "days": 2, "highlights": []}
    assert scoring.score_trip(TRIP, bad).deterministic == pytest.approx(0.0)


def test_trip_without_a_fixed_destination_or_length_skips_those_checks() -> None:
    open_trip = example(
        {"destination": None, "days": None, "highlights": [], "criteria": "c"}
    )
    scored = scoring.score_trip(
        open_trip, {"destination": "x", "days": 9, "highlights": ["a"]}
    )
    assert set(scored.parts) == {"highlights"}


EXPENSE = example(
    {
        "amount_minor": 12050,
        "currency": "PLN",
        "payer_name": "Ania",
        "included_names": [],
        "excluded_names": ["Marka"],
        "description": "obiad",
    }
)


def test_expense_checks_amount_currency_and_names() -> None:
    answer = {
        "amount_minor": 12050,
        "currency": "PLN",
        "payer_name": "ania",
        "included_names": [],
        "excluded_names": ["Marka"],
        "description": "kolacja",
    }
    scored = scoring.score_expense(EXPENSE, answer)
    assert scored.deterministic == pytest.approx(1.0)
    wrong = answer | {"amount_minor": 1205, "currency": None}
    assert scoring.score_expense(EXPENSE, wrong).deterministic == pytest.approx(0.6)


RECEIPT = example(
    {
        "amount_minor": 1000,
        "currency": "PLN",
        "spent_on": "2026-08-14",
        "merchant": "Sklep",
        "items": [
            {"name": "a", "amount_minor": 400},
            {"name": "b", "amount_minor": 600},
        ],
    }
)


def test_receipt_compares_totals_dates_and_line_totals() -> None:
    answer = {
        "amount_minor": 1000,
        "currency": "PLN",
        "spent_on": "2026-08-14",
        "merchant": "Sklep Spożywczy",
        "items": [
            {"name": "x", "amount_minor": 600},
            {"name": "y", "amount_minor": 400},
        ],
    }
    assert scoring.score_receipt(RECEIPT, answer).deterministic == pytest.approx(1.0)
    wrong = answer | {"spent_on": "2026-14-08", "items": []}
    assert scoring.score_receipt(RECEIPT, wrong).deterministic == pytest.approx(0.6)


def test_match_is_right_only_for_the_expected_id() -> None:
    sample = example({"place_id": "p1"})
    assert scoring.score_match(
        sample, {"place_id": "p1"}
    ).deterministic == pytest.approx(1.0)
    assert scoring.score_match(
        sample, {"place_id": None}
    ).deterministic == pytest.approx(0.0)
    none = example({"place_id": None})
    assert scoring.score_match(none, {"place_id": None}).deterministic == pytest.approx(
        1.0
    )


def test_combine_mixes_the_judge_by_weight() -> None:
    scored = Scored(deterministic=1.0, parts={}, judge=None, judge_weight=0.0)
    assert scoring.combine(scored, None) == pytest.approx(1.0)
    mixed = Scored(deterministic=1.0, judge_weight=0.25)
    assert scoring.combine(mixed, 0.0) == pytest.approx(0.75)
