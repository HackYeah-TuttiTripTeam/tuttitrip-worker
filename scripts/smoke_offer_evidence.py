"""Real-model smoke test of the offer evidence agents (never runs in CI).

Runs the quote extractor and the decision-model judge outside DBOS on a short
German/Polish/English offer, applies the same pure verification as the
workflow and prints quote, verdict and confidence per requirement. Keys come
from the environment (``TUTTITRIP_LLM__GB10_API_KEY`` and/or
``OPENROUTER_API_KEY``). Nothing secret is printed.

    uv run python scripts/smoke_offer_evidence.py
"""

import asyncio
import sys

from tuttitrip_worker.accommodation.agents import extract_quotes, judge_quote
from tuttitrip_worker.accommodation.logic.evidence import (
    Assessment,
    build_evidence,
    verified_quotes,
)
from tuttitrip_worker.contracts import RequirementLabel

OFFER = """\
Wohnung in Berlin-Mitte, 2 Schlafzimmer, 65 m2.
Ausstattung: WLAN kostenlos, voll ausgestattete Küche, Waschmaschine.
Parkplatz in der Tiefgarage gegen Aufpreis (15 EUR pro Nacht).
Brak basenu na terenie obiektu.
The sauna on the roof terrace is open in summer only.
Haustiere sind nicht erlaubt.
"""
LABELS = [
    RequirementLabel(key="wifi", label="Wi-Fi"),
    RequirementLabel(key="pool", label="Basen"),
    RequirementLabel(key="parking", label="Bezpłatny parking"),
    RequirementLabel(key="sauna", label="Sauna"),
    RequirementLabel(key="pets_allowed", label="Zwierzęta dozwolone"),
    RequirementLabel(key="gym", label="Siłownia"),
]


async def main() -> int:
    """Run the pipeline once and print the evidence.

    Returns:
        Process exit code: 0 when every step answered, 1 otherwise.
    """
    keys = [label.key for label in LABELS]
    extracted = await extract_quotes(OFFER, keys, LABELS)
    quotes = verified_quotes(OFFER, extracted, keys)
    named = {label.key: label.label for label in LABELS}
    assessments: dict[tuple[str, str], Assessment] = {}
    unanswered = 0
    for key in keys:
        for text in quotes[key]:
            judged = await judge_quote(key, named[key], text)
            if judged is None:
                unanswered += 1
            else:
                assessments[key, text] = judged
    for item in build_evidence(keys, quotes, assessments):
        if not item.quotes:
            print(f"{item.requirement_key}: no quote (unconfirmed)")
        for quote in item.quotes:
            print(
                f"{item.requirement_key}: {quote.verdict} "
                f"({quote.confidence}) {quote.text!r}"
            )
    return 1 if unanswered else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
