# tuttitrip-worker

Worker w tle dla TuttiTrip, planera wyjazdów grupowych i rodzinnych
budowanego na HackYeah 2026. Wykonuje operacje, które trwają za długo na
zwykłe zapytanie HTTP: uruchomienia agentów Pydantic AI (wywiad,
uzasadnienia, parsowanie wklejonych planów, oferty, paragony), liczenie
embeddingów do pgvector, wzbogacanie danych o miejscach i przeliczanie planów.

Każde takie zadanie jest workflowem [DBOS](https://docs.dbos.dev/python).
DBOS zapisuje postęp w Postgresie. Jeśli worker padnie albo zostanie
zrestartowany w trakcie, po starcie kończy zadanie od ostatniego zapisanego
kroku, więc wywołania modelu, które już się udały, nie są powtarzane.

Backend ([tuttitrip-backend](https://github.com/HackYeah-TuttiTripTeam/tuttitrip-backend))
tylko wrzuca zadania do kolejki i odczytuje ich stan, wynik oraz postęp.
Workflowy, kolejki i kontrakt należą do tego repo.

Solver i inne reguły deterministyczne nie zależą od LLM: agenci przygotowują
szkice, decyzje podejmuje czysty kod, a testy architektury tego pilnują.

![Plansza „Gdzie pracuje AI”: sześć kart (wywiad głosem i tekstem, profil osoby z jednego zdania, nagłe zdarzenia w podróży, uzasadnienie planu, TuttiTrip w ChatGPT i Claude.ai, mapy i kalendarz) ze statusem i nazwami modeli, pod spodem pasek o Pydantic AI.](docs/readme/13-ai.webp)

> [!NOTE]
> Opis całego projektu, plansze, zrzuty aplikacji i instrukcja uruchomienia wszystkich części są w repozytorium zbiorczym [tuttitrip](https://github.com/HackYeah-TuttiTripTeam/tuttitrip). TuttiTrip powstał z pomocą asystentów kodowania (Claude Code, Codex). Ludzie z zespołu odpowiadali za architekturę rozwiązania, rozplanowanie funkcji, działanie aplikacji i to, jak się z niej korzysta.

## Wymagania

- [uv](https://docs.astral.sh/uv/) i Python 3.14 (uv sam go pobierze),
- Docker,
- lokalnie uruchomiony backend: jego Postgres (z pgvector) i migracje,
- opcjonalnie [Ollama](https://ollama.com) z `nomic-embed-text` do embeddingów
  oraz klucz OpenRouter do agentów. Testy nie potrzebują ani jednego, ani drugiego.

## Uruchomienie lokalne razem z backendem

Schemat bazy, migracje i tabele DBOS należą do backendu, więc najpierw
uruchamiamy jego część:

```bash
cd ../tuttitrip-backend
docker compose up -d --wait db                 # Postgres 18 + pgvector
uv run alembic upgrade head                    # tabele aplikacji
uv run dbos migrate -s postgresql://tuttitrip:tuttitrip@localhost:5432/tuttitrip
uv run uvicorn tuttitrip.main:app --reload     # API na :8000
```

Potem worker, na hoście:

```bash
cd ../tuttitrip-worker
uv sync
cp .env.example .env        # domyślne wartości pasują do compose backendu
uv run tuttitrip-worker
```

Albo w kontenerze podłączonym do sieci compose backendu (`tuttitrip_default`):

```bash
docker compose up --build
```

W logach powinny pojawić się `DBOS launched!`, `Listening to 3 queues` oraz
`worker ready: env=local`. Całą ścieżkę, od backendu przez kolejkę do workera,
sprawdzisz tak:

```bash
curl -X POST localhost:8000/api/v1/jobs/ping          # zwraca workflow_id
curl localhost:8000/api/v1/jobs/ping/<workflow_id>    # status SUCCESS
```

Jeśli Postgres backendu działa na innym porcie (np. 5433), zmień
`TUTTITRIP_WORKER_DATABASE_URL` w `.env`. Embeddingi wymagają `ollama pull
nomic-embed-text`. Agent korzysta z `OPENROUTER_API_KEY` albo z lokalnego
endpointu zgodnego z OpenAI (`TUTTITRIP_LLM__LOCAL_*`).

## Testy, lint, typy

```bash
uv run ruff check .          # lint (select = ALL, preview)
uv run ruff format --check . # formatowanie
uv run ty check              # typy, tryb ścisły
uv run pytest -m "not integration and not e2e"  # testy jednostkowe i architektury, to samo robi CI
uv run pytest -m integration            # lokalnie przed PR: runtime DBOS na SQLite (CI ich nie uruchamia)
uv run pytest                                    # wszystko; Postgres ani sieć nie są potrzebne
```

Testy oznaczone `integration` (lokalnie, CI ich nie uruchamia) uruchamiają prawdziwy runtime DBOS na tymczasowym pliku SQLite i
wrzucają zadania przez `DBOSClient` tak samo jak backend.
Modele zastępują `TestModel` i `FunctionModel` z Pydantic AI, a
`ALLOW_MODEL_REQUESTS = False` blokuje każde prawdziwe wywołanie. Kroki
zapisujące do Postgresa podmieniamy w testach workflowów, a ich SQL
sprawdzamy osobno.

CI uruchamia te same cztery komendy i dodatkowo sprawdza, czy
`contracts/jobs.schema.json` jest aktualny.

## Architektura

![Diagram architektury w czterech kolumnach: ludzie, aplikacja, backend oraz worker i modele, połączone kropkowanymi liniami, pod nim lista technologii.](docs/readme/14-stack.webp)

Dwa przykłady pracy workera (plansze na danych przykładowych):

<table>
  <tr>
    <td width="50%"><img width="100%" src="docs/readme/07-settle.webp" alt="Telefon z listą wydatków wyjazdu, obok suma 586 zł, „3 przelewy zamiast 6” i niepewny odczyt paragonu z przerywanym obrysem."><br><sub>Model odczytuje paragon albo wpis wydatku, a niepewny odczyt czeka na potwierdzenie. Saldo i przelewy liczy kod.</sub></td>
    <td width="50%"><img width="100%" src="docs/readme/06-replan.webp" alt="Telefon z planem dnia po komunikacie „Silny deszcz od 11:00”: muzea zamiast parku i molo, obok panel „Co sprawdził kod” z zerem problemów."><br><sub>Nagłe zdarzenie: model zamienia „deszcz od 11:00” na warunki, solver przelicza resztę dnia (w budowie).</sub></td>
  </tr>
</table>

Kod jest podzielony na pionowe plastry (vertical slices):

```text
src/tuttitrip_worker/
  contracts.py   kontrakt z backendem: nazwy, payloady, wersja (czysty moduł)
  main.py        rejestracja workflowów, DBOS.launch(), obsługa SIGTERM
  shared/        config, dbos (kolejki), llm (OpenRouter, model lokalny), db
  system/        ping (smoke test) i heartbeat co 30 s
  planning/      trwały agent planujący (generate_trip_plan)
  linter/        parse_pasted_plan: wklejony plan do pozycji z cytatem
  embeddings/    embed_texts → pgvector; logic/ to czysta logika
```

W domenie mamy `workflows.py` (deterministyczna orkiestracja), `steps.py`
(całe I/O: modele, HTTP, baza), `schemas.py`, `agents.py` (agenci
Pydantic AI z `DBOSDurability`) i opcjonalnie `logic/` z czystym kodem.
Testy pytest-archon pilnują, żeby `shared` nie importował domen, domeny nie
sięgały do swoich wnętrz nawzajem, a moduły czyste (`contracts.py`, każdy
`schemas.py` i `logic/`) nie dotykały Pydantic AI, DBOS ani bazy, także
pośrednio. Pełne zasady, w tym pułapki DBOS, opisuje [AGENTS.md](AGENTS.md).

Kolejki:

| Kolejka | Do czego | Limit |
| --- | --- | --- |
| `default` | ping, embeddingi, wzbogacanie, przeliczenia | 8 naraz |
| `local_llm` | model lokalny na GPU (dellpromaxgb10) | 2 naraz |
| `openrouter` | modele przez OpenRouter | 8 naraz, 30 startów na minutę |

## Integracja z backendem

Wspólne zasady obu repozytoriów są w `deploy/CONVENTIONS.md` w repo backendu
(sekcja „Integracja z workerem”). W skrócie:

- Kontrakt. Źródłem prawdy jest `src/tuttitrip_worker/contracts.py`:
  nazwy workflowów i kolejek, modele wejścia, wyjścia i zdarzeń oraz
  `CONTRACT_VERSION`. Backend ma lustro w
  `src/tuttitrip/shared/jobs/contracts.py`. Oba repo generują ten sam plik
  `contracts/jobs.schema.json` (`uv run python scripts/export_contracts.py`),
  a job `contracts-check` w CI porównuje go z plikiem z drugiego repo.
- Wersjonowanie. Każdy payload ma `contract_version`. Worker odrzuca
  wersję, której nie obsługuje, czytelnym błędem `ContractError`, a backend
  pokazuje go w `GET /api/v1/jobs/{id}`. Przy niezgodnej zmianie worker najpierw
  przyjmuje starą i nową wersję, potem backend przełącza się na nową, a na
  końcu worker porzuca starą.
- Dane. Schemat i migracje należą do backendu. Worker nie wykonuje DDL
  i łączy się jako rola `tuttitrip_worker`: czyta tabele domenowe, a pisze
  tylko do `embeddings`, `job_results` i `worker_heartbeats`. Payloady są
  małe (identyfikatory i parametry), resztę danych worker pobiera z bazy.
- Wyniki. Mały wynik jest wyjściem workflowu. Duże albo trwałe wyniki
  trafiają do `job_results` lub do tabel domenowych. Postęp idzie przez
  zdarzenie `progress`. Worker nigdy nie woła backendu po HTTP.
- Idempotencja. Backend nadaje deterministyczne `workflow_id`, a kroki
  workera robią upsert (np. id embeddingu to UUIDv5 ze źródła, modelu i tekstu).
- Heartbeat. Co 30 s worker zapisuje się w `worker_heartbeats`. Backend
  pokazuje to w `/api/v1/health` i nie przyjmuje zadań, gdy workera brakuje.
- Smoke test. Po wdrożeniu worker woła `POST /api/v1/jobs/ping` na API swojego
  środowiska i czeka na `SUCCESS`. Brak wyniku oznacza nieudany deploy.
- Wersja aplikacji i serializacja. `application_version` to nazwa
  środowiska (`main`, `develop`, slug gałęzi), stała po obu stronach, bo
  worker pobiera z kolejki tylko zadania ze swoją wersją. Argumenty i wyniki
  idą jako przenośny JSON (`PORTABLE`); domyślny pickle wymagałby tych
  samych klas Pythona po obu stronach.

Nowe zadanie dodajemy skillem `new-workflow`, a zmianę kontraktu
synchronizujemy skillem `sync-contracts` (oba w `.claude/skills/`).

## Wdrożenie

Każdy push uruchamia CI raz: `lint` i `tests` równolegle na runnerach
`[self-hosted, hackathon]`, `contracts-check` też na tych runnerach. Podgląd
gałęzi wdraża się od razu, `main` i `develop` czekają na zielone `lint` i
`tests`. Deploy idzie na runnerze zainstalowanym na `dellpromaxgb10`
(`[self-hosted, tuttitrip-worker-deploy]`).

| Gałąź | Obraz | Kontener |
| --- | --- | --- |
| `main` | `tuttitrip-worker:main` | `tuttitrip-worker-main` |
| `develop` | `tuttitrip-worker:develop` | `tuttitrip-worker-develop` |
| inna | `tuttitrip-worker:<slug>` | `tuttitrip-worker-<slug>` |

Slug powstaje tak samo jak w backendzie: małe litery, a każdy ciąg znaków
spoza `[a-z0-9]` zamienia się na `-` (`feature/cos tam` → `feature-cos-tam`).

Co robi deploy (`deploy/deploy.sh`):

1. Buduje obraz `tuttitrip-worker:<env>`.
2. Jeśli środowisko backendu `<env>` istnieje, (re)startuje kontener
   `tuttitrip-worker-<env>` w sieci `tuttitrip`, z plikiem
   `~/tuttitrip/envs/<env>.worker.env` (zapisuje go deploy backendu) oraz
   `~/tuttitrip/worker.env` (sekrety workera). Zastępuje przy tym workera
   zapasowego, którego backend mógł wcześniej uruchomić z `:develop` lub `:main`.
3. Jeśli środowiska backendu nie ma (gałąź istnieje tylko w workerze),
   deploy tylko buduje obraz i niczego nie uruchamia. Kontener wystartuje
   przy pierwszym deployu backendu na gałęzi o tej samej nazwie.
4. Czeka na healthcheck i uruchamia smoke test przez API.
5. Sprząta: usuwa obrazy gałęzi, których nie ma już w tym repo. Jeśli
   środowisko backendu takiej gałęzi wciąż działa, przełącza jego workera
   na obraz zapasowy (`:develop`, potem `:main`), żeby nie zostało bez
   workera. `main` i `develop` nie są nigdy usuwane.

Gdy backend wdraża środowisko X i nie ma kontenera `tuttitrip-worker-X`,
uruchamia go sam z `tuttitrip-worker:X`, a jeśli takiego obrazu nie ma, to
z `:develop`, a w ostateczności z `:main`.

Sekrety: `OPENROUTER_API_KEY` jest sekretem GitHub Actions, który deploy
dopisuje do `~/tuttitrip/worker.env` na hoście. Embeddingi na hoście liczy
Ollama (`nomic-embed-text`), do której worker ma dostęp przez sieć `ollama_net`.

## Git flow

- `main` to produkcja, `develop` to integracja. Na obu wymagany jest PR i
  zielone `lint` i `tests`, bez force-push i bez usuwania (o ile plan GitHuba na to
  pozwala).
- Gałęzie zakładamy od `develop`: `feature/<nazwa>`, `fix/<nazwa>`,
  `chore/<nazwa>`. PR idzie do `develop`, a wydanie to PR `develop` → `main`.
- Po merge'u gałąź usuwa workflow `Delete merged branch` i uruchamia
  sprzątanie jej obrazu. `main` i `develop` nie są nigdy usuwane, więc PR
  wydania idzie prosto z `develop`. Automatyczne usuwanie gałęzi w
  ustawieniach GitHuba jest wyłączone, bo kasowało `develop` po wydaniu.
- Zmiana kontraktu trafia najpierw tutaj, a potem do lustra w backendzie.
