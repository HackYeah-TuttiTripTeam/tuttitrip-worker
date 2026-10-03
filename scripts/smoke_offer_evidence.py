"""Real-model smoke test of the offer evidence agents (never runs in CI).

Runs the workflow's pipeline (``assess_offer``) outside DBOS on a short
German/Polish/English offer and prints quote, verdict and confidence per
requirement. Keys come from the environment (``TUTTITRIP_LLM__GB10_API_KEY`` and/or
``OPENROUTER_API_KEY``). Nothing secret is printed.

    uv run python scripts/smoke_offer_evidence.py
"""

import asyncio
import sys

from tuttitrip_worker.accommodation.services.assess import assess_offer
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
        Process exit code: 0 when every requirement with a quote was assessed.
    """
    evidence = await assess_offer(OFFER, [label.key for label in LABELS], LABELS)
    unassessed = 0
    for item in evidence:
        if not item.quotes:
            print(f"{item.requirement_key}: no quote (unconfirmed)")
        for quote in item.quotes:
            unassessed += quote.verdict is None
            print(
                f"{item.requirement_key}: {quote.verdict} "
                f"({quote.confidence}) {quote.text!r}"
            )
    return 1 if unassessed else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
