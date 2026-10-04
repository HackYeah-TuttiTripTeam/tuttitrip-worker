"""Examples derived from other examples (pure)."""

from collections.abc import Sequence
from typing import Any

from tuttitrip_worker.bench.schemas import Example


def verdict_examples(offers: Sequence[Example]) -> list[Example]:
    """One judge-stage example per requirement that has reference quotes.

    The stage-two question ("what does this quote say about the requirement?")
    needs no new authoring: the quotes and verdicts are in the offer examples.

    Args:
        offers: Examples of ``extract_offer_evidence``.

    Returns:
        Examples with ``input = {key, label, quotes}`` and
        ``expected = {verdicts}``; ids are ``<offer id>#<key>``.
    """
    derived: list[Example] = []
    for offer in offers:
        labels: dict[str, Any] = {
            item["key"]: item.get("label") for item in offer.input["requirements"]
        }
        for key, entry in offer.expected["evidence"].items():
            quotes = entry["quotes"]
            if not quotes:
                continue
            derived.append(
                Example(
                    id=f"{offer.id}#{key}",
                    lang=offer.lang,
                    tags=[*offer.tags, "derived"],
                    input={
                        "key": key,
                        "label": labels.get(key),
                        "quotes": [q["text"] for q in quotes],
                    },
                    expected={"verdicts": [q["verdict"] for q in quotes]},
                    notes=offer.notes,
                )
            )
    return derived
