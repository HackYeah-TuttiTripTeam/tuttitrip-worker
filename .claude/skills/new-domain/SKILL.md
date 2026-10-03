---
name: new-domain
description: Create a new vertical-slice domain (or subdomain) package in tuttitrip-worker with the required files and passing architecture tests. Use when a new area of background work does not fit system, planning or embeddings.
---

# Add a domain

1. Create `src/tuttitrip_worker/<domain>/` (snake_case) with:
   - `__init__.py`: a one-line docstring, **no imports** (test-enforced);
   - `workflows.py`: the domain's `@DBOS.workflow`s (see `new-workflow`);
   - `schemas.py`: internal Pydantic models (pure: stdlib + pydantic only);
   - optional `steps.py` (I/O), `agents.py` (Pydantic AI), `logic/`
     (pure code, needs `__init__.py` + at least one module), `services/`.
   A subdomain is a nested package with the same layout.
2. Dependencies (AGENTS.md "Dependency rules"): import shared code from
   `tuttitrip_worker.shared.*` and `tuttitrip_worker.contracts`; another
   domain only through its `schemas`. Move anything two domains need into
   `shared/`. `pydantic_ai` only in `agents.py`, `dbos` only in
   `workflows.py`/`steps.py`, SQL only in `steps.py`.
3. Import the domain's workflows in `src/tuttitrip_worker/main.py` (they must
   be registered before `DBOS.launch()`).
4. Add `tests/domains/test_<domain>.py`.
5. Run `uv run pytest tests/architecture` and then all four checks.
