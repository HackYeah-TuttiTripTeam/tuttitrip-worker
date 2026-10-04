# Benchmark modeli LLM

Ręczny pomiar trafności, opóźnienia i kosztu modeli katalogu `shared/llm` na
zestawach wzorcowych (golden sets). Nie działa w CI (kosztuje i potrzebuje sieci).
Wyniki przebiegów: `llm-<data>.md` w tym katalogu, z surowymi wynikami w `llm-<data>.jsonl`.

## Uruchomienie

```bash
# klucze tylko ze środowiska albo z pliku, który proces czyta sam (nie trafiają do env ani do logów)
uv run python -m tuttitrip_worker.bench --case parse_expense_text --models chat,openrouter \
  --env-file ~/.config/tuttitrip/keys.env

uv run python -m tuttitrip_worker.bench --case all                     # wszystkie przypadki i trasy
uv run python -m tuttitrip_worker.bench --case all --limit 3 --no-report  # szybka próba
```

| Opcja | Znaczenie |
| --- | --- |
| `--case` | nazwy po przecinku albo `all` |
| `--models` | trasy katalogu bez `tuttitrip:`; domyślnie `agent,chat,decide,decide-laya,decide-cloud,openrouter` |
| `--env-file` | plik z `GB10_LITELLM_KEY`, `OPENROUTER_API_KEY` i opcjonalnie `ANTHROPIC_API_KEY` |
| `--judge` | model sędziego (domyślnie `claude-sonnet-5-5`); nazwa ze `/` to slug OpenRoutera |
| `--no-judge` | tylko sprawdzenia deterministyczne |
| `--limit N` | tylko pierwsze N przykładów każdego przypadku |
| `--concurrency N` | równoległe wywołania na model (domyślnie 2; do czystego pomiaru opóźnienia użyj 1) |
| `--report PATH`, `--no-report` | gdzie zapisać raport (domyślnie `docs/benchmarks/llm-<data>.md`) |

Klucze: `TUTTITRIP_LLM__GB10_API_KEY` albo `GB10_LITELLM_KEY` (Qwen, basal, Laya),
`OPENROUTER_API_KEY` (referencja i JEV), `ANTHROPIC_API_KEY` (sędzia). Trasa bez klucza
jest pomijana (`skipped`), więc można mierzyć część modeli.

Z workflowu: nie ma go celowo. Jeśli kiedyś będzie potrzebny, ma to być
`workflow_dispatch` z kluczami jako sekretami, nigdy krok zwykłego pipeline'u
(test `test_ci_never_runs_the_benchmark` pilnuje, że CI tego nie wywołuje).

## Co jest mierzone

Każdy przypadek woła tę samą funkcję, co worker (te same instrukcje, walidatory
wyjścia i ponowienia), z podmienionym modelem. Bierzemy pierwszy model trasy bez
zapasowych, żeby awaria głównego modelu nie ukryła się za fallbackiem.

| Przypadek | Co modele robią |
| --- | --- |
| `extract_offer_evidence` | cytaty z oferty noclegu i werdykty dla wymagań (cały potok; tylko modele językowe) |
| `offer_verdict` | sam etap sędziego oferty: werdykt jednego cytatu (modele decyzyjne i językowe; przykłady wyprowadzone z poprzedniego przypadku) |
| `parse_pasted_plan` | pozycje planu wklejonego z czatbota |
| `generate_trip_plan` | szkic planu z prośby |
| `parse_expense_text` | wydatek z jednego zdania |
| `read_receipt` | paragon albo zrzut z banku (obraz) |
| `match_places` | które miejsce katalogu ma na myśli pozycja planu (modele decyzyjne i językowe) |

Modele decyzyjne (basal, Laya, JEV) odpowiadają tylko wyborem jednej opcji, więc na
przypadkach z wolnym tekstem mają `skipped`.

### Ocena przykładu

1. Sprawdzenia deterministyczne (kod): kwoty w groszach, waluty, daty, identyfikatory
   miejsc, werdykty, stany wymagań, dopasowanie pozycji planu po nazwie i ich pola.
   Dają `deterministic` (0 do 1).
2. Sędzia LLM ocenia tylko to, czego kod nie sprawdzi (opis wydatku, nazwy pozycji
   paragonu, adresy w planie, trafność cytatów, dopasowanie atrakcji do prośby), według
   rubryki `tests/golden/rubric.md` (plus `<przypadek>/rubric.md`), skalą 0, 0.25, 0.5,
   0.75, 1 z krótkim uzasadnieniem. Ocena końcowa to średnia ważona (waga sędziego jest
   w `bench/constants.py`: od 0,1 do 0,6 zależnie od przypadku).
3. Wyjście bez poprawnej struktury po ponowieniach to `invalid` i ocena 0. Awaria
   dostawcy (sieć, limit, timeout) to `error`: nie wlicza się do trafności, tylko do
   odsetka błędów dostawcy.

Sędzia: z `ANTHROPIC_API_KEY` wołamy `claude-sonnet-5-5` na zgodnym z OpenAI
endpoincie Anthropic (bez dodatkowej biblioteki); bez niego ten sam model przez OpenRouter
(`anthropic/claude-sonnet-5.5`). Model i adres są w ustawieniach `TUTTITRIP_BENCH__*`.
Ścieżka bezpośrednio do Anthropic nie była uruchamiana w przebiegach z tego repozytorium
(brak klucza), przebiegi szły przez OpenRouter.

### Koszt

Z tokenów wszystkich prób (`bench/constants.py`, `PRICES`). GB10 (Qwen, basal, Laya) to
własny sprzęt: 0 USD, bez energii i amortyzacji. Ceny OpenRouter z dnia przebiegu.
Cena JEV 1.13 nie jest opublikowana, więc jej koszt to `n/d`.

## Zestawy wzorcowe

`tests/golden/<przypadek>/examples.jsonl`, opis formatu i zasad w
[tests/golden/README.md](../../tests/golden/README.md). Każdy ma co najmniej 20 przykładów
po polsku i 5 po angielsku z trudnymi przypadkami (wstrzyknięcie instrukcji, brak danych,
obce waluty, sprzeczne cytaty, imiona w odmianie). Dane są zmyślone, bez danych osobowych.
Test `tests/bench/test_golden.py` pilnuje kształtu i spójności.

## Dodanie przypadku

1. Katalog `tests/golden/<nazwa>/` z `examples.jsonl` (i opcjonalnym `rubric.md`).
2. Funkcja `run_*` w `bench/cases.py` (woła funkcję produkcyjną przez `subject.using(agent)`),
   scorer w `bench/logic/scoring.py` i wpis w `CASES`.
3. Test scorera w `tests/bench/test_scoring.py`.
