"""The golden sets hold up to the rules in tests/golden/README.md."""

import json
from pathlib import Path

import pytest

from tuttitrip_worker.bench.cases import CASES
from tuttitrip_worker.bench.cli import REPO_ROOT, examples_of
from tuttitrip_worker.bench.golden import load_examples, load_rubric
from tuttitrip_worker.bench.schemas import Example
from tuttitrip_worker.quotes import find_quote

GOLDEN = REPO_ROOT / "tests/golden"
MIN_PL = 20
MIN_EN = 5
DIRECTORIES = sorted(p.name for p in GOLDEN.iterdir() if p.is_dir())


def test_every_case_has_a_golden_set() -> None:
    sources = {case.source or case.name for case in CASES.values()}
    assert sources == set(DIRECTORIES)


@pytest.mark.parametrize("directory", DIRECTORIES)
def test_set_is_big_enough_and_ids_are_unique(directory: str) -> None:
    examples = load_examples(GOLDEN, directory)
    ids = [e.id for e in examples]
    assert len(ids) == len(set(ids))
    assert sum(e.lang == "pl" for e in examples) >= MIN_PL
    assert sum(e.lang == "en" for e in examples) >= MIN_EN
    assert all(e.id.startswith(e.lang) for e in examples)


@pytest.mark.parametrize("name", list(CASES))
def test_examples_of_a_case_load(name: str) -> None:
    assert examples_of(CASES[name], GOLDEN, None)
    assert load_rubric(GOLDEN, CASES[name].source or name)


def test_offer_quotes_are_verbatim_spans_of_the_offer() -> None:
    for example in load_examples(GOLDEN, "extract_offer_evidence"):
        offer = example.input["offer"]
        keys = [r["key"] for r in example.input["requirements"]]
        assert set(example.expected["evidence"]) == set(keys)
        for entry in example.expected["evidence"].values():
            for quote in entry["quotes"]:
                assert find_quote(offer, quote["text"]) == quote["text"], example.id


def test_plan_places_are_written_in_the_text() -> None:
    for example in load_examples(GOLDEN, "parse_pasted_plan"):
        for item in example.expected["items"]:
            assert item["place_name"] in example.input["text"], example.id


def test_match_answers_are_in_the_catalog() -> None:
    for example in load_examples(GOLDEN, "match_places"):
        ids = {p["place_id"] for p in example.input["places"]}
        answer = example.expected["place_id"]
        assert answer is None or answer in ids, example.id
        assert example.input["name"] in example.input["quote"], example.id


def test_receipt_images_exist_and_the_totals_add_up() -> None:
    directory = GOLDEN / "read_receipt"
    for example in load_examples(GOLDEN, "read_receipt"):
        assert (directory / example.input["image"]).is_file(), example.id
        items = example.expected["items"]
        total = example.expected["amount_minor"]
        if items and total is not None:
            assert sum(i["amount_minor"] for i in items) == total, example.id


def test_expense_amounts_are_none_or_positive() -> None:
    for example in load_examples(GOLDEN, "parse_expense_text"):
        amount = example.expected["amount_minor"]
        assert amount is None or amount > 0, example.id


def test_examples_are_one_json_object_per_line() -> None:
    for directory in DIRECTORIES:
        path: Path = GOLDEN / directory / "examples.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            assert isinstance(json.loads(line), dict)
            Example.model_validate_json(line)
