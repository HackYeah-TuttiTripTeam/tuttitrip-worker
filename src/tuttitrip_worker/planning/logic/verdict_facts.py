"""Verdict facts from a stored plan version (pure).

The backend stores the result of a plan version (``plan_versions.result``,
``PlanRead`` of ``docs/algorytm.md``, section 10) with ``verdicts``,
``explain`` and the fairness ledger. Here the parts a justification may use
are cut out of it, with names instead of profile ids.
"""

from collections.abc import Mapping
from itertools import batched
from typing import Any

from tuttitrip_worker.planning.schemas import (
    PersonUtility,
    SidedPerson,
    VerdictFacts,
)


def _names(result: Mapping[str, Any]) -> dict[str, str]:
    ledger = (result.get("fairness") or {}).get("per_person") or []
    return {row["profile_id"]: row["name"] for row in ledger}


def _place_names(result: Mapping[str, Any]) -> dict[str, str]:
    return {
        stop["place_id"]: stop["name"]
        for day in result.get("days") or []
        for stop in day.get("items") or []
    }


def _sided(
    entries: list[dict[str, Any]], names: Mapping[str, str]
) -> list[SidedPerson]:
    return [
        SidedPerson(
            name=names[entry["profile_id"]], reason_code=entry.get("reason_code")
        )
        for entry in entries
        if entry["profile_id"] in names
    ]


def build_facts(result: Mapping[str, Any]) -> list[VerdictFacts]:
    """Facts for every verdict of a plan version result.

    Args:
        result: ``plan_versions.result`` of the backend.

    Returns:
        One entry per verdict, in the stored order; empty when the plan has
        no verdicts yet.
    """
    names = _names(result)
    place_names = _place_names(result)
    explain: dict[str, list[PersonUtility]] = {}
    for entry in result.get("explain") or []:
        if entry["profile_id"] in names:
            explain.setdefault(entry["place_id"], []).append(
                PersonUtility(
                    name=names[entry["profile_id"]],
                    match=entry["match"],
                    effort=entry["effort"],
                    utility=entry["utility"],
                )
            )
    return [
        VerdictFacts(
            place_id=verdict["place_id"],
            place_name=place_names.get(verdict["place_id"]),
            verdict=verdict["verdict"],
            v_p=verdict.get("v_p"),
            yes=_sided(verdict.get("yes") or [], names),
            no=_sided(verdict.get("no") or [], names),
            skip_codes=verdict.get("skip_codes") or [],
            substitute_name=place_names.get(verdict.get("substitute_place_id") or ""),
            people=explain.get(verdict["place_id"], []),
        )
        for verdict in result.get("verdicts") or []
    ]


def split_batches(facts: list[VerdictFacts], size: int) -> list[list[VerdictFacts]]:
    """Cut the facts into batches for separate model calls.

    Args:
        facts: All verdict facts of the plan.
        size: Largest batch.

    Returns:
        Consecutive batches; none is empty.
    """
    return [list(batch) for batch in batched(facts, size, strict=False)]
