"""Quote and assess an offer: the pipeline the workflow and the smoke script share."""

from tuttitrip_worker.accommodation import agents
from tuttitrip_worker.accommodation.logic.evidence import (
    Assessment,
    build_evidence,
    verified_quotes,
)
from tuttitrip_worker.contracts import RequirementEvidence, RequirementLabel


async def assess_offer(
    offer: str, keys: list[str], labels: list[RequirementLabel]
) -> list[RequirementEvidence]:
    """Find verbatim quotes for each requirement and assess them.

    The extractor (language model) proposes quotes, pure code keeps only the
    verbatim ones, and the decision model assesses all quotes of a requirement
    in one call. Meaning of the result, per requirement:

    * ``quotes == []``: the offer is silent (or nothing checked out);
    * ``verdict is None`` on a quote: the judge was unavailable;
    * several quotes may disagree (present and absent): the worker does not
      aggregate, the backend shows such a requirement as conflicting.

    Only requirement keys with a text signal are meant to be sent (``pool``,
    ``parking``); keys without one (platform, distance to attractions) are the
    backend's to decide, and keys whose meaning is not obvious need a label.

    Args:
        offer: The pasted offer text.
        keys: Unique requirement keys (``ExtractOfferEvidenceInput`` checks it).
        labels: Human labels of some keys.

    Returns:
        One ``RequirementEvidence`` per key, in the order of ``keys``.
    """
    extracted = await agents.extract_quotes(offer, keys, labels)
    quotes = verified_quotes(offer, extracted, keys)
    named = {item.key: item.label for item in labels}
    assessments: dict[tuple[str, str], Assessment] = {}
    for key in keys:
        if not quotes[key]:
            continue
        judged = await agents.judge_quotes(key, named.get(key), quotes[key])
        if judged is not None:
            for text, assessment in zip(quotes[key], judged, strict=True):
                assessments[key, text] = assessment
    return build_evidence(keys, quotes, assessments)
