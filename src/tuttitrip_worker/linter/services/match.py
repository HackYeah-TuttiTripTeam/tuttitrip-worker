"""Match parsed items to catalog places: the pipeline the workflow uses."""

from tuttitrip_worker.contracts import ParsedPlanItem, PlaceMatch
from tuttitrip_worker.linter import agents, steps
from tuttitrip_worker.linter.logic.candidates import (
    match_with_pick,
    match_without_model,
)
from tuttitrip_worker.linter.schemas import RankedPlace
from tuttitrip_worker.shared.config.settings import get_settings


async def match_items(
    items: list[ParsedPlanItem], city_slug: str, seed: str
) -> list[PlaceMatch]:
    """Match each item to a catalog place of the city, or leave it unrecognized.

    Pure code ranks the candidates; the decision model only picks one of them
    or "none". Without candidates no model is asked. When the model is switched
    off (``TUTTITRIP_LINTER__MATCH_WITH_MODEL=false``) or no model can answer,
    the similarity threshold decides. An item never disappears: the linter
    counts an unrecognized one as a place without data.

    Args:
        items: Items that passed the quote checks.
        city_slug: City slug from the payload.
        seed: Stable per-job value (the document id).

    Returns:
        One ``PlaceMatch`` per item, in the order of ``items``.
    """
    names = list(dict.fromkeys(item.place_name for item in items))
    if not names:
        return []
    by_name = await steps.rank_candidates(city_slug, names)
    use_model = get_settings().linter.match_with_model
    matches: list[PlaceMatch] = []
    for item in items:
        ranked = [RankedPlace.model_validate(p) for p in by_name[item.place_name]]
        if not ranked or not use_model:
            matches.append(match_without_model(item.index, ranked))
            continue
        choice = await agents.choose_place(item.place_name, item.quote, seed, ranked)
        if choice is None:
            matches.append(match_without_model(item.index, ranked))
        else:
            matches.append(match_with_pick(item.index, ranked, *choice))
    return matches
