# AGENTS.md

Canonical instructions for anyone (human or agent) changing this repository.
`CLAUDE.md` imports this file. The user-facing docs are in `README.md` (Polish).

## What this is

`tuttitrip-worker` is the background worker of TuttiTrip (HackYeah 2026, a
group/family trip planner). It runs the long operations as durable
[DBOS](https://docs.dbos.dev/python) workflows on Postgres: Pydantic AI agent
runs (interview, justifications, parsing pasted plans, offers, receipts),
embeddings into pgvector, enrichment of place data and plan recomputation.

The backend, [tuttitrip-backend](https://github.com/HackYeah-TuttiTripTeam/tuttitrip-backend)
(FastAPI), never runs workflows. It only enqueues them and reads their status,
results and events through `DBOSClient`. This repo owns every workflow, queue
and the job contract. The rules shared by both repos live in the backend's
`deploy/CONVENTIONS.md` ("Integracja z workerem"); change them there first.

Product rule: deterministic logic (solver, linter, pricing) never depends on
an LLM, on DBOS or on the database. Agents only draft; pure code decides.
The architecture tests enforce it.

## Commands

```bash
uv sync                                   # install (incl. dev group)
cp .env.example .env                      # local settings
uv run tuttitrip-worker                   # run the worker (needs Postgres, see README)
docker compose up --build                 # the worker in a container, on the backend's compose network
uv run python scripts/export_contracts.py # regenerate contracts/jobs.schema.json

# Must all pass before every commit (CI runs the same):
uv run ruff check . && uv run ruff format --check . && uv run ty check && uv run pytest
```

## Layout: vertical slices

```text
src/tuttitrip_worker/
  contracts.py         canonical job contract (names, payloads, version); pure
  main.py              composition root: WORKFLOWS, SCHEDULES, run() (launch + SIGTERM)
  healthcheck.py       Docker HEALTHCHECK (liveness file written by main)
  shared/              shared kernel, imports no domain
    config/            Settings (pydantic-settings)
    dbos/              DBOS config, queues + limits, report_progress()
    llm/               model catalog (OpenRouter, local) and embedder
    db/                async engine, table mappings (no DDL), job_results step
  system/              ping (smoke test) + heartbeat schedule
  planning/            durable planner agent (generate_trip_plan)
  embeddings/          embed_texts -> pgvector; logic/ = pure row building
src/tuttitrip_dbos_dashboard/  read-only DBOS dashboard (not a worker domain), see below
contracts/jobs.schema.json   rendered contract, compared with the backend mirror
deploy/                host deployment scripts (bash)
scripts/               export_contracts.py
tests/architecture/    structure + dependency rules (pytest-archon)
```

### Files in a domain or subdomain

| File / package | Required | Contents |
| --- | --- | --- |
| `__init__.py` | yes | Docstring only. **No imports** (test-enforced). |
| `workflows.py` | yes | `@DBOS.workflow` functions. Deterministic orchestration only. |
| `schemas.py` | yes | Internal Pydantic models. Pure. |
| `steps.py` | if I/O | `@DBOS.step` functions: model calls, HTTP, database. |
| `agents.py` | if LLM | Module-level Pydantic AI agents with `DBOSDurability`. |
| `logic/` | optional | Pure logic (rules, math, row building). |
| `services/` | optional | Larger I/O helpers used by steps. |
| `<subdomain>/` | optional | Same layout, nested. |

Anything else in a domain directory fails `test_domain_contains_only_known_files`.

### Dependency rules (tests/architecture)

1. `shared` never imports a domain (transitive).
2. A domain never imports another domain, except its `schemas` (direct).
   Shared code goes to `shared/`.
3. **Pure modules** = `tuttitrip_worker.contracts`, every `schemas.py` and every
   module in a `logic/` package. They must not reach `pydantic_ai`, `dbos`,
   `sqlalchemy`, `psycopg`, `pgvector`, `openai` or `httpx`, even
   transitively, nor `workflows`/`steps`/`agents`/`services`/`shared`/`main`.
   Identified by module path (regex), so adding a `logic/` package or a
   `schemas.py` opts it in automatically.
4. Only `agents.py` and `shared.llm` import `pydantic_ai`.
5. Only `workflows.py`, `steps.py`, `shared.dbos`, `shared.db` and `main`
   import `dbos`.
6. Only `steps.py` and `shared.db` import `sqlalchemy`/`psycopg`/`pgvector`.

pytest-archon 0.0.7 notes: `should_not_import` is transitive by default,
`only_direct_imports=True` limits it, imports in functions and
`TYPE_CHECKING` blocks count, parent `__init__.py` execution is not modelled
(hence import-free `__init__.py`), and a rule matching nothing fails.

## DBOS rules (verified against dbos 3.2 docs/source and pydantic-ai 2.54)

1. **Workflows are deterministic.** A workflow body may validate, branch and
   call steps; it must not do I/O, read clocks or random numbers. Every model
   call, HTTP request and database query goes into a `@DBOS.step` (retries:
   `retries_allowed=True`). Step outputs are checkpointed and replayed on
   recovery.
2. **Async workflows use the `_async` APIs.** Inside `async def` workflows
   call `await DBOS.set_event_async(...)`; the sync variants raise
   "called while an event loop is running".
3. **Durable agents.** `Agent(..., capabilities=[DBOSDurability()])`, created
   at module level, with a unique `name=` (it prefixes the step names), and
   `await agent.run()` called **inside** a `@DBOS.workflow`. Outside a workflow
   the capability is transparent (and not durable). `DBOSAgent` is deprecated
   (removed in pydantic-ai v3). Tools that do I/O need their own `@DBOS.step`.
4. **Models cross step boundaries as strings.** A `Model` instance cannot be
   serialized into a step; unregistered instances are rejected. Agents use the
   ids `tuttitrip:openrouter` / `tuttitrip:local` (`shared/llm/models.py`) and
   the catalog's `ResolveModelId` capability builds the real model inside the
   step from settings. Tests swap models with `catalog.override(TestModel())`.
5. **Register before launch.** Every workflow, step and agent must exist
   before `DBOS.launch()`: `main.py` imports all workflow modules at import
   time. Queues are registered after launch with `DBOS.register_queue`
   (dbos >= 3.2; the old `Queue(...)` constructor is gone), schedules with
   `DBOS.apply_schedules` (idempotent).
6. **application_version = `<env>`, constant.** A worker only dequeues
   workflows tagged with its own `application_version`, and version-less
   workflows go only to the latest registered version. Both sides use the env
   name: the worker reads `DBOS__APPVERSION` (set by the backend deploy in
   `envs/<env>.worker.env`, default `local`) and passes it explicitly in
   `DBOSConfig`; the backend enqueues with `app_version=<env>`. Trade-off: a
   redeployed worker resumes in-flight workflows with its new code, so keep
   workflow step order compatible or drain queues before risky changes.
7. **Serialization: portable JSON.** `DBOSClient.enqueue` pickles arguments
   by default, which would need the backend's Python classes here. Every
   backend-facing workflow is declared with
   `serialization_type=WorkflowSerializationFormat.PORTABLE`, takes one JSON
   object and returns one; events are set with `serialization_type=PORTABLE`.
   Errors become portable `{name, message, code, data}`
   (`PortableWorkflowError` on the backend side).
8. **Queues** (`shared/dbos/runtime.py`): `default` (worker concurrency 8:
   ping, embeddings, enrichment, recomputation), `local_llm` (concurrency 2:
   the GPU on dellpromaxgb10), `openrouter` (concurrency 8, at most 30 starts
   per 60 s across workers). A queue is polled only by the application that
   registered it (`tuttitrip-worker`). Queues live in the system database
   (`dbos.queues`), so a name removed from `Queue` keeps being polled until
   its row is gone: `register_queues()` deletes this application's queues
   that are not in the contract, unless they still hold queued work.
9. **Shutdown and recovery.** SIGTERM/SIGINT stop the main loop;
   `DBOS.destroy(workflow_completion_timeout_sec=30)` lets running workflows
   finish, then the process exits (`--stop-timeout 40` on the container).
   Anything still running stays `PENDING` and is recovered on the next start
   (same executor id `local`, same application version). Steps can run again
   after a crash, so they must be idempotent (upserts, deterministic ids).
10. **No DDL.** `run_migrations: False`: the backend creates the `dbos`
    schema with `dbos migrate -r tuttitrip_worker`. Tests turn migrations on
    only for their throwaway SQLite file.
11. **Progress** for the frontend: `await report_progress(stage, percent)`
    sets the `progress` event; the backend reads it with `get_event`.

## Contract and integration with the backend

- `contracts.py` is canonical: `APPLICATION_NAME`, `CONTRACT_VERSION`,
  `SUPPORTED_CONTRACT_VERSIONS`, `Queue`, `Workflow`, payload models,
  `Progress`, `WORKFLOWS` (default queue per workflow), `queue_for(provider)`.
  The backend mirror is `src/tuttitrip/shared/jobs/contracts.py` in
  tuttitrip-backend.
- Both repos render the contract with the same algorithm (`contract_json()`:
  JSON Schemas without titles/descriptions, sorted keys) and commit it as
  `contracts/jobs.schema.json`. `tests/test_contracts.py` checks the file is
  current. CI job `contracts-check` diffs it against the backend's file (same
  branch, else develop, else main) and reports drift as a warning; the
  backend's `contracts-check` fails on drift. Both need a read token secret
  (`BACKEND_REPO_TOKEN` here, `WORKER_REPO_TOKEN` there) and skip without it.
- Every payload carries `contract_version`. `parse_input()` rejects an
  unsupported version or an invalid payload with `ContractError`
  (`code`, `data`); the workflow ends `ERROR` and the backend shows the
  message in `GET /api/v1/jobs/{id}`. Unknown fields are ignored (additive changes).
- **Incompatible change procedure:** (1) worker: add the new version to
  `SUPPORTED_CONTRACT_VERSIONS`, accept both payload shapes, deploy (the
  heartbeat advertises `min_contract_version`..`contract_version`);
  (2) backend: bump `CONTRACT_VERSION` and its models; (3) worker: drop the
  old version. Use the `sync-contracts` skill.
- **Data.** The backend owns the schema and migrations. The worker connects
  as role `tuttitrip_worker` (`TUTTITRIP_WORKER_DATABASE_URL`) with SELECT on
  the domain tables it reads (`trips`, `profiles`) and write access only to
  `embeddings`, `job_results`, `worker_heartbeats`. `shared/db/tables.py`
  maps their columns without DDL; `tests/test_no_ddl.py` forbids
  `create_all`/DDL. A new table = backend migration + `deploy/worker-grants.sql`
  there, then a mapping here.
- **Results.** Small results are the workflow output. Large or persistent
  ones go to `job_results` (`save_job_result` step) or domain tables
  (`embeddings`). Payloads stay small (ids + parameters); fetch data by id in
  a step. The worker never calls the backend over HTTP.
- **Idempotency.** The backend enqueues with a deterministic workflow id
  (`return-existing`), so a repeated request returns the same job. Steps
  upsert: `embeddings.id` is a UUIDv5 of (source, model, text), `job_results`
  is keyed by workflow id, heartbeats by worker id.
- **Heartbeat.** The scheduled `heartbeat` workflow (every 30 s) upserts
  `worker_heartbeats(worker_id=tuttitrip-worker-<env>, env, contract_version,
  min_contract_version, app_version, last_seen)`. Backend `/api/v1/health` and its
  enqueue endpoints use it.
- **Smoke test.** `ping` (echo, no LLM). After a worker deploy of env X,
  `deploy/smoke.sh` calls `POST https://tuttitrip-api[-X].gburek.app/api/v1/jobs/ping`
  and polls `GET /api/v1/jobs/ping/{id}` until `SUCCESS` (~3 min max), else the deploy fails.

| Workflow | Queue | Input | Output |
| --- | --- | --- | --- |
| `generate_trip_plan` | `openrouter` (or `local_llm` for `provider=local`) | `{contract_version, trip_id, request, provider}` | `{contract_version, destination, days, highlights}` |
| `embed_texts` | `default` | `{contract_version, source_kind, source_id, texts}` | `{contract_version, model, dimensions, stored}` |
| `ping` | `default` | `{contract_version, message}` | `{contract_version, message, worker_app_version}` |

## Conventions

- Ruff `select = ["ALL"]` with preview. Google docstrings on every public
  module, class and function, `Args:` when there are parameters and
  `Returns:`/`Yields:` when something comes back. Ignores live in
  `pyproject.toml` with a reason. Fix code instead of suppressing.
- ty in strict mode (`all = "error"`, warnings fail).
- pytest strict, warnings are errors. Tests never call real models
  (`models.ALLOW_MODEL_REQUESTS = False`) and never need Postgres: the `dbos`
  fixture launches DBOS on a throwaway SQLite file and the `client` fixture
  enqueues through `DBOSClient` exactly like the backend. Steps that touch
  Postgres are replaced with `monkeypatch` in workflow tests; their SQL is
  compiled and asserted separately.
- Domain tests go in `tests/domains/`, shared infrastructure in `tests/shared/`.

## Settings and secrets

- `shared/config/settings.py`: prefix `TUTTITRIP_`, nested delimiter `__`,
  plus the standard names `DBOS_SYSTEM_DATABASE_URL` and `DBOS__APPVERSION`.
  `.env.example` must list exactly the Settings fields (`tests/test_settings.py`).
- `OPENROUTER_API_KEY` is read by Pydantic AI under its standard name unless
  `TUTTITRIP_LLM__OPENROUTER_API_KEY` is set.
- Never commit secrets or `.env`, never print them. CI/deploy secrets are
  GitHub Actions secrets; host-only settings live in `~/tuttitrip/worker.env`.

## Design system

Skill `tuttitrip-design-system` (`.claude/skills/tuttitrip-design-system`) jest wspólny dla wszystkich
repozytoriów TuttiTrip; UI powstaje we frontendzie. Teksty pisane przez modele w workerze (uzasadnienia
werdyktów, podsumowania, pytania) mają ton i słownik z README skilla: po polsku albo angielsku według
języka użytkownika, na „Ty”, krótko, bez emoji, z nazwami z słownika („sprawdzenie planu”, werdykty
„Obowiązkowo”, „Pasuje”, „Kultowe, ale nie Twoje”, „Pomiń”). Wpisz te zasady w instrukcje agentów.

## Praca agentów nad issues

Nad backlogiem pracuje równolegle kilku agentów AI i ludzi. Te zasady pilnują, żeby nikt nie wchodził
innym w drogę i żeby każda funkcja przeszła ten sam proces. Dotyczą też ludzi.

1. Wybór issue. Bierzesz tylko issue z tablicy
   [TuttiTrip](https://github.com/orgs/HackYeah-TuttiTripTeam/projects/1) ze statusem Todo, bez etykiety
   `in-progress` i bez przypisanej osoby. Linia „Zależy od:” w opisie wymienia issues, które muszą być
   zmergowane do `develop`. Jeśli któreś nie jest, pracuj tylko na jego kontrakcie (np. stała odpowiedź z
   OpenAPI) i napisz to w komentarzu. Kolejność: najpierw P0, potem P1, w obrębie milestone'u.
2. Zajęcie issue, zanim napiszesz kod:
   - `gh issue edit <nr> --add-label in-progress`,
   - Status na tablicy: In Progress,
   - komentarz „Start” z nazwą gałęzi, ścieżką worktree i krótkim planem (pliki, które zmienisz).
   Etykieta `in-progress` znaczy „zajęte”. Nie bierz takiego issue i nie zmieniaj go bez zgody zespołu.
3. Worktree i gałąź. Nigdy nie pracuj w głównym klonie repozytorium. Jedno issue to jeden worktree, jedna
   gałąź i jeden PR do `develop`:

   ```bash
   git -C ~/Documents/GitHub/<repo> fetch origin
   git -C ~/Documents/GitHub/<repo> worktree add -b feature/<nr>-<krotka-nazwa> \
     ~/Documents/GitHub/worktrees/tuttitrip/<repo>-<nr>-<krotka-nazwa> origin/develop
   ```

   (`<repo>` to `tuttitrip-backend`, `tuttitrip-worker` albo `tuttitrip-frontend`; w repo zbiorczym
   `tuttitrip` gałąź bierzesz z `origin/main`.)
4. Komentarze ze statusem w issue po każdym etapie: plan, implementacja z testami, wynik smoke testu,
   wynik review subagenta, link do PR. Krótko: co zrobione, co dalej, co blokuje. Gdy utkniesz: etykieta
   `blocked` i komentarz z powodem i tym, czego potrzebujesz.
5. Pliki wspólne, w których łatwo o konflikt, zmieniaj małymi krokami i przed PR rób
   `git fetch origin && git rebase origin/develop`:
   - `src/tuttitrip_worker/contracts.py` i `contracts/jobs.schema.json` (najpierw tu, potem lustro w
     backendzie, skill `sync-contracts`),
   - `main.py` (rejestracja workflowów), `shared/llm/models.py` (katalog modeli), `shared/dbos/runtime.py`.
6. Smoke test jest obowiązkowy dla KAŻDEGO zrealizowanego feature'a. Po pushu gałęzi poczekaj na
   wdrożenie podglądu i przejdź na żywo scenariusz z kryteriów akceptacji issue:
   - worker gałęzi startuje tylko obok wdrożenia backendu o tej samej nazwie gałęzi (deploy/CONVENTIONS.md
     w backendzie); bez niego uruchom backend i worker lokalnie (README),
   - zleć zadanie przez API (`POST /api/v1/...` z backendu) i odpytuj `GET /api/v1/jobs/{id}` aż do
     `SUCCESS`; przy błędzie sprawdź kroki workflow w panelu DBOS albo logach kontenera.
   Wynik (kroki, odpowiedzi albo zrzuty ekranu) wpisz w komentarzu w issue. Bez zielonego smoke testu
   nie ma PR.
7. Review subagenta. Po zielonym smoke teście uruchom subagenta-recenzenta z diffem gałęzi, treścią
   issue i story źródłową. Sprawdza:
   - uproszczenie kodu i zbędną złożoność (skille `simplify` i `ponytail-review`),
   - złożoność logiki,
   - poprawność biznesową względem story, słownika z dokumentu architektonicznego i, przy logice
     planowania, specyfikacji algorytmu (`docs/algorytm.md` w tuttitrip-backend).
   Popraw to, co znalazł, i **powtórz smoke test**. Wynik review i drugiego smoke testu wpisz w komentarzu.
8. PR. Dopiero po tym otwórz PR do `develop` skillem `open-pr` (`Closes #<nr>`) i ustaw Status: In
   Review. Po merge'u zdejmij `in-progress`, usuń worktree
   (`git -C ~/Documents/GitHub/<repo> worktree remove <ścieżka>`); issue zamyka `Closes`, Status: Done.

## Zgłoszenia, PR i wydania

Zasady są wspólne dla całej organizacji, pełny opis jest w
[CONTRIBUTING.md](https://github.com/HackYeah-TuttiTripTeam/.github/blob/main/CONTRIBUTING.md).

Zgłoszenia (issues):

- Tytuł zaczyna się od `feat:`, `docs:`, `chore:` albo `bug:`, opcjonalnie
  z zakresem, np. `feat(worker): Eksport planu do PDF`. Regex:
  `^(feat|docs|chore|bug)(\([a-z0-9-]+\))?: \S.{3,}`.
- Treść ma sekcje `###` i żadna wymagana nie może być pusta. W `feat`,
  `docs` i `chore` są to Opis, Dlaczego, Kryteria akceptacji, Definition of
  Done i Obszar (w `feat` można dodać Poza zakresem). W `bug` są to Opis,
  Kroki do odtworzenia, Oczekiwane zachowanie, Faktyczne zachowanie,
  Środowisko, Dlaczego, Kryteria akceptacji, Definition of Done i Obszar.
- `.github/workflows/issue-format.yml` sprawdza każde nowe i edytowane
  zgłoszenie. Złe zamyka jako "not planned", dodaje etykietę
  `invalid-format` i pisze w komentarzu, co poprawić. Po poprawce otwiera je
  ponownie. Ustawia też etykietę `type:*`.
- Z terminala (skill `new-issue` przygotuje treść i założy zgłoszenie):

  ```bash
  gh issue create --title "feat(worker): Eksport planu do PDF" --body-file - <<'MD'
  ### Opis
  Organizator pobiera gotowy plan jako PDF.

  ### Dlaczego
  W podróży plan musi być dostępny offline, a nie każdy instaluje PWA.

  ### Kryteria akceptacji
  - [ ] Given gotowy plan, When kliknę "Pobierz PDF", Then dostanę plik z planem dzień po dniu

  ### Definition of Done
  - [ ] CI zielone (lint, typy, testy, testy architektury)
  - [ ] PR zmergowany do `develop` i sprawdzony na wdrożeniu develop

  ### Obszar
  Worker
  MD
  ```

Pull requesty i merge:

- Tytuł PR: `feat:`, `docs:`, `chore:` albo `bugfix:` (w PR nie `bug:`),
  opcjonalnie z zakresem. PR wydania `develop` -> `main` ma tytuł
  `release: opis`.
- Opis po polsku według szablonu: `## Co i dlaczego`, `## Powiązane issue`
  (`Closes #12` albo `Refs #12`; w `docs` i `chore` może być `brak`),
  `## Lista zmian`, `## Jak przetestować`, `## Zrzuty ekranu`,
  `## Checklista`. Gotowy szablon ma skill `open-pr`.
- `.github/workflows/pr-format.yml` oznacza check na czerwono i komentuje,
  gdy tytuł albo sekcje są złe. Bez ochrony gałęzi (darmowy plan) czerwony
  check nie blokuje merge'a, więc nie mergujemy z czerwonym.
- PR do `develop` mergujemy przez "Squash and merge" (tytuł PR staje się
  commitem). PR wydania do `main` mergujemy przez "Create a merge commit".
  GitHub nie pozwala ustawić metody osobno dla gałęzi, więc to zasada
  zespołu.

Wydania:

- `.github/workflows/release-notes.yml` (Release Drafter, konfiguracja w
  repozytorium `.github`) nadaje PR etykietę `type:*` według prefiksu tytułu
  i po każdym merge'u do `develop` aktualizuje szkic następnego wydania w
  GitHub Releases.
- Merge PR `release:` do `main` publikuje szkic i zakłada tag `vX.Y.Z`.
  `feat` podnosi wersję minor, pozostałe typy patch, pierwsze wydanie to
  `v0.1.0`. Nie prowadzimy pliku CHANGELOG.md.

## Git flow

- `main` is production, `develop` is integration; both protected (PR + green
  `checks`, no force-push, no deletion) where the GitHub plan allows it.
- Branch from `develop`: `feature/<short-name>`, `fix/<short-name>`,
  `chore/<short-name>`. PR into `develop`; release = PR `develop` -> `main`.
- After a merge the `Delete merged branch` workflow
  (`.github/workflows/delete-merged-branch.yml`, logic in the org `.github`
  repo) deletes the head branch and starts `cleanup.yml` for its image. It
  never deletes `main` or `develop`, so release PRs go straight from
  `develop`. GitHub's "Automatically delete head branches" is off: without
  branch protection it deleted `develop` after every release PR.
- A contract change lands here first, then in the backend mirror.
- No AI attribution in commits, PRs or docs (no `Co-Authored-By` trailers for
  assistants, no "generated with" lines).

## Deployment

Every push runs CI (`checks` and `contracts-check` on `[self-hosted, hackathon]`),
then `deploy` on the runner installed on the host
(`[self-hosted, tuttitrip-worker-deploy]`, in `~/tuttitrip-worker-runner`).

| Branch | Image | Container | Backend env file |
| --- | --- | --- | --- |
| `main` | `tuttitrip-worker:main` | `tuttitrip-worker-main` | `~/tuttitrip/envs/main.worker.env` |
| `develop` | `tuttitrip-worker:develop` | `tuttitrip-worker-develop` | `~/tuttitrip/envs/develop.worker.env` |
| any other | `tuttitrip-worker:<slug>` | `tuttitrip-worker-<slug>` | `~/tuttitrip/envs/<slug>.worker.env` |

Slug rules are the backend's (`deploy/lib.sh`, `tests/test_deploy_naming.py`).

`deploy/deploy.sh <branch> <sha>`:

1. Builds `tuttitrip-worker:<env>` (labels `tuttitrip.managed=true`,
   `tuttitrip.role=worker`, `tuttitrip.env`, branch, sha).
2. Takes `flock ~/tuttitrip/deploy.lock` (shared with the backend deploy).
3. If the backend env exists (`envs/<env>.worker.env`), (re)starts
   `tuttitrip-worker-<env>` on network `tuttitrip`, `--restart unless-stopped`,
   `--stop-timeout 40`, env files `envs/<env>.worker.env` + `~/tuttitrip/worker.env`,
   and also attaches it to `ollama_net` (embeddings via `http://ollama:11434/v1`).
   This replaces a develop/main fallback worker the backend may have started.
4. If the backend env does not exist (worker-only feature branch), it only
   builds the image. The next backend deploy of that env starts it.
5. Waits for the Docker healthcheck, runs `deploy/smoke.sh <env>`.
6. Runs `deploy/cleanup.sh`.

Fallbacks: when the backend deploys env X and no `tuttitrip-worker-X` runs, it
starts one from `tuttitrip-worker:X`, else `:develop`, else `:main` (backend's job).

Cleanup (every deploy + on branch deletion): images `tuttitrip-worker:<env>`
whose branch no longer exists in this repo are removed. A container running
such an image is switched to the fallback (`:develop`, then `:main`) if its
backend env still exists, otherwise removed. `main`/`develop` and anything
without our name prefix and labels are never touched. The backend cleanup
removes the worker container when the backend env itself goes away.

`~/tuttitrip/worker.env` (mode 600) is created by the first deploy with host
defaults; existing keys are never overwritten, so hand edits survive.
`OPENROUTER_API_KEY` comes from the GitHub secret of the same name.

## DBOS dashboard (`tuttitrip_dbos_dashboard`)

There is no DBOS Conductor or Console here. Self-hosted Conductor is under a
proprietary license: a trial key is only for development and evaluation,
production needs a paid license, and the free key accepts one executor per
application, while we run main and develop under the same application name.
Instead, a small read-only dashboard ships in this repository and image.

- https://tuttitrip-dbos.gburek.app shows the queues with their backlog, the
  newest workflows (filters for status, name and limit) and the steps and
  error of a single workflow, for main and develop.
- It is stdlib `http.server` plus the official `DBOSClient` (`list_workflows`,
  `list_queued_workflows`, `list_queues`, `list_workflow_steps`). It never
  loads inputs or outputs and never writes; DBOSClient runs no migrations.
- Login: the backend's admin stack (`deploy/admin/` in tuttitrip-backend) puts
  it behind nginx `auth_request` + oauth2-proxy (Auth0, superadmin
  allow-list). It is reachable only on the private `tuttitrip-admin` network.
- It connects as role `tuttitrip_readonly`
  (`TUTTITRIP_DBOS_DASHBOARD_DATABASES=main=postgresql://...,develop=...`,
  written by the backend's `deploy/admin/setup.sh`). The URLs are never
  rendered or logged.
- It is a separate top-level package, so the worker's architecture rules do
  not apply to it. It may import `dbos` and must stay read-only. Its tests are
  in `tests/dashboard/`, including a round trip through a real DBOS on SQLite.
- Deploy: entry point `tuttitrip-dbos-dashboard` in `tuttitrip-worker:main`,
  container `tuttitrip-dbos-dashboard` (compose project `tuttitrip-admin`).
  After building main, `deploy/deploy.sh` restarts it on the new image when
  `~/tuttitrip/admin/compose.yaml` exists.

## Powiadomienia (Discord)

- `.github/workflows/discord-notify.yml` wysyła na Discord zespołu wynik
  każdego innego workflow tego repozytorium (`workflow_run: completed`) przez
  wspólny `discord-notify.yml` z repozytorium
  [`.github`](https://github.com/HackYeah-TuttiTripTeam/.github). GitHub
  uruchamia go tylko z kopii na `main`, ta na `develop` jest dla porządku.
- Nowy albo przemianowany workflow trzeba dopisać po nazwie (`name:`) do
  listy `workflows:` w tym pliku.
- Zasady szumu:
  - `skipped` nie idzie wcale,
  - `Issue format`, `PR format`, `Delete merged branch`, `Release notes` i sprzątanie (`Cleanup branch deployments`) piszą tylko przy niepowodzeniu,
  - sukces na `main` i `develop` to pełna wiadomość,
  - sukces na innej gałęzi i anulowanie to jedna linia,
  - błąd to zawsze pełna wiadomość z listą nieudanych jobów.
- Zmiany w projekcie #1 oraz nowe, zamknięte i scalone issue i PR wysyła
  Worker `tuttitrip-discord-relay` z webhooka organizacji (kod w
  `.github/discord-relay`), nie ten workflow.
- Webhook to sekret repozytorium `DISCORD_WEBHOOK_URL` (sekret organizacji
  nie działa: na darmowym planie nie widzą go repozytoria prywatne). Rotacja:
  nowy webhook w Discordzie, `gh secret set DISCORD_WEBHOOK_URL` w czterech
  repozytoriach i `wrangler secret put DISCORD_WEBHOOK_URL` w Workerze.
  Szczegóły w
  [CONTRIBUTING.md](https://github.com/HackYeah-TuttiTripTeam/.github/blob/main/CONTRIBUTING.md#powiadomienia-discord).
  URL-a webhooka nie wklejamy nigdzie (issue, PR, logi, czat).
