---
name: new-workflow
description: Add a DBOS workflow to tuttitrip-worker end to end - contract entry, workflow and steps, optional durable agent, tests, exported schema - and remind about the backend mirror. Use when adding a background job (agent run, embedding, enrichment, recomputation) or a new workflow the backend should enqueue.
---

# Add a workflow

Read AGENTS.md "DBOS rules" first. Pick the domain (or run `new-domain`).

1. **Contract** (`src/tuttitrip_worker/contracts.py`):
   - add a `Workflow` member (snake_case name, it is the DBOS name);
   - add `<Name>Input(ContractPayload)` / `<Name>Output(ContractPayload)`.
     Inputs are small: ids and parameters, never whole documents. Use
     `Literal[...]` instead of enums in payloads (keeps the schema inline);
   - add the `WORKFLOWS` entry with its default `Queue` (`default` for non-LLM
     work, `openrouter` / `local_llm` for LLM work, see `queue_for`).
   - New optional field = compatible. Anything else = new contract version:
     follow "Incompatible change procedure" in AGENTS.md.
2. **Steps** (`<domain>/steps.py`): every model call, HTTP request or DB
   query is an `@DBOS.step(retries_allowed=True, ...)`, idempotent (upsert,
   deterministic ids). DB writes only to tables granted to `tuttitrip_worker`
   (`shared/db/tables.py`); a new table needs a backend migration + grant first.
3. **Agent** (if LLM, `<domain>/agents.py`): module-level
   `Agent(model_id(LlmProvider.OPENROUTER), name="<unique>", output_type=...,
   defer_model_check=True, capabilities=[catalog.capability(), DBOSDurability()])`.
4. **Workflow** (`<domain>/workflows.py`):

   ```python
   @DBOS.workflow(name=Workflow.X.value, serialization_type=PORTABLE)
   async def x(payload: dict[str, Any]) -> dict[str, Any]:
       request = parse_input(XInput, payload)  # version + validation
       await report_progress("working", 10)
       ...  # steps / agent.run(model=model_id(...))
       return XOutput(...).model_dump(mode="json")
   ```

   Deterministic body only; `await DBOS.*_async` APIs inside async workflows.
5. **Wire it**: add it to `WORKFLOWS` in `src/tuttitrip_worker/main.py`.
6. **Tests** (`tests/domains/test_<domain>.py`): enqueue with
   `tests.helpers.enqueue(client, dbos, Workflow.X, payload)`; override models
   with `catalog.override(TestModel(...))` / `FunctionModel`, embeddings with
   `embedder.override(TestEmbeddingModel(...))`; replace Postgres steps with
   `monkeypatch.setattr(steps, "<step>", fake)` and assert their SQL compiled
   with `postgresql.dialect()`. `test_contracts.py` already checks the new
   name is registered in DBOS.
7. Regenerate and check:
   `uv run python scripts/export_contracts.py`, then
   `uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest`.
8. **Backend mirror (do not skip):** open a PR in tuttitrip-backend that
   mirrors the change in `src/tuttitrip/shared/jobs/contracts.py`, adds a
   `TIMEOUT_SECONDS` entry and regenerates its `contracts/jobs.schema.json`
   (their `sync-contracts` skill). Until then this repo's `contracts-check`
   warns and the backend's fails.
