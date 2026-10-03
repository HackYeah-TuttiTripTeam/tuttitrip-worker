"""Real-model smoke test of the pasted-plan parser (never runs in CI).

Runs ``parser_agent`` (outside DBOS, so no database is needed) on the test
fixture plan (``tests/linter_sample.py``) with a prompt-injection line added,
and checks with the workflow's own pure code that every fixture quote is
contained in a kept quote, in order, that nothing is unread and that the
injection did not empty the plan. Keys come from the environment
(``TUTTITRIP_LLM__GB10_API_KEY`` or ``GB10_LITELLM_KEY``, and
``OPENROUTER_API_KEY``). Nothing secret is printed.

    PYTHONPATH=. uv run python scripts/smoke_parse_plan.py

Workflow-level check, once the backend can enqueue ``parse_pasted_plan``:
POST a ``plan`` document (``/planning/linter/trips/{trip}/documents``), enqueue
the job with that ``document_id`` through the backend, poll ``GET /jobs/{id}``
until ``SUCCESS`` and compare ``items[].quote`` with the document text.
"""

import asyncio
import os
import sys
import time

from pydantic import SecretStr
from tests.linter_sample import INJECTION, PLAN, QUOTES

from tuttitrip_worker.linter.agents import parser_agent, read_pasted_plan
from tuttitrip_worker.linter.logic.quotes import split_items
from tuttitrip_worker.shared.config.settings import LlmSettings
from tuttitrip_worker.shared.llm.models import ModelKey, build_model


async def main() -> int:
    """Parse the fixture plan with the real chat model chain.

    Returns:
        Process exit code: 0 when the kept quotes are the expected ones.
    """
    gb10_key = os.environ.get("TUTTITRIP_LLM__GB10_API_KEY") or os.environ.get(
        "GB10_LITELLM_KEY", ""
    )
    settings = LlmSettings(gb10_api_key=SecretStr(gb10_key))
    text = PLAN.replace("\n\n", f"\n\n{INJECTION}\n", 1)
    started = time.monotonic()
    with parser_agent.override(model=build_model(ModelKey.CHAT, settings)):
        draft = await read_pasted_plan(text, "smoke", "krakow")
    items, unread = split_items(draft.items, text)
    print(
        f"{len(items)} items, {len(unread)} unread, {time.monotonic() - started:.1f}s"
    )
    for item in items:
        print(
            f"  {item.index} d{item.day} {item.start_time}-{item.end_time} "
            f"{item.place_name} | {item.amount_minor} {item.currency} "
            f"| {item.transport}"
        )
    for bad in unread:
        print(f"  unread ({bad.reason}): {bad.quote!r}")
    # The model may quote a little more of the line than the fixture does.
    ok = (
        len(items) == len(QUOTES)
        and all(want in item.quote for item, want in zip(items, QUOTES, strict=True))
        and not unread
    )
    print("ok" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
