"""Real-endpoint smoke test of the model catalog (never runs in CI).

Calls every leg of the catalog (Qwen agent and chat, OpenRouter, basal, JEV and,
with ``--laya``, Laya) with a tiny Polish prompt, one by one, so a working
fallback cannot hide a broken primary model. Keys come from the environment:
``TUTTITRIP_LLM__GB10_API_KEY`` (or ``GB10_LITELLM_KEY``) and
``OPENROUTER_API_KEY``. Nothing secret is printed.

    uv run python scripts/smoke_models.py [--laya]
"""

import asyncio
import os
import sys
import time
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.fallback import FallbackModel

from tuttitrip_worker.shared.config.settings import LlmSettings
from tuttitrip_worker.shared.llm.models import ModelKey, build_model


class City(BaseModel):
    """Structured answer for the language models."""

    name: str
    country: str


class Reason(BaseModel):
    """Pick-one answer for the decision models."""

    kind: Literal["transport", "lodging", "food"]


def legs(key: ModelKey, settings: LlmSettings) -> list[Model]:
    """Split a fallback chain into the models it is made of.

    Args:
        key: Catalog id.
        settings: LLM settings.

    Returns:
        The single model, or every model of a ``FallbackModel`` in order.
    """
    model = build_model(key, settings)
    return list(model.models) if isinstance(model, FallbackModel) else [model]


async def check(label: str, model: Model, *, decision: bool) -> bool:
    """Run one tiny request on one model and report the outcome.

    Args:
        label: Name shown in the report.
        model: The model to call.
        decision: Ask a pick-one question instead of a structured city.

    Returns:
        Whether the model answered.
    """
    started = time.monotonic()
    agent = Agent(model, output_type=Reason if decision else City)
    prompt = "Bilet na pociąg do Krakowa" if decision else "Miasto z Wawelem?"
    try:
        result = await agent.run(prompt)
    except Exception as error:  # ruff: ignore[blind-except] - the smoke test reports every failure
        print(f"FAIL {label}: {type(error).__name__}: {str(error)[:200]}")
        return False
    print(f"ok   {label}: {result.output} ({time.monotonic() - started:.1f}s)")
    return True


async def main(*, with_laya: bool) -> int:
    """Smoke every catalog leg.

    Args:
        with_laya: Also call Laya (its endpoint is not always deployed).

    Returns:
        Process exit code: 0 when every call answered.
    """
    gb10_key = os.environ.get("GB10_LITELLM_KEY", "")
    settings = LlmSettings(gb10_api_key=gb10_key) if gb10_key else LlmSettings()
    plan = [
        (ModelKey.AGENT, False),
        (ModelKey.CHAT, False),
        (ModelKey.DECIDE, True),
        (ModelKey.DECIDE_CLOUD, True),
    ]
    if with_laya:
        plan.append((ModelKey.DECIDE_LAYA, True))
    results: list[bool] = []
    for key, decision in plan:
        for model in legs(key, settings):
            label = f"{key.value} -> {model.model_name}"
            results.append(await check(label, model, decision=decision))
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(with_laya="--laya" in sys.argv)))
