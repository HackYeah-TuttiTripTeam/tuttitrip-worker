"""Markdown report of a run (pure): tables of models by cases."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from tuttitrip_worker.bench.constants import CALLS_PER_COST_UNIT, PASS_SCORE
from tuttitrip_worker.bench.schemas import Summary

NO_DATA = "n/d"
"""Cell of a combination that did not run or has no value."""

RECOMMENDATIONS_HEADING = "## Rekomendacje"
"""Heading of the section the maintainers write by hand after a run."""

INTRO = (
    "Wynik ręcznego przebiegu `uv run python -m tuttitrip_worker.bench`. "
    "Opis metody, rubryka sędziego i sposób uruchomienia: [README.md](README.md)."
)
METHOD_ACCURACY = (
    "- Trafność: średnia ocena 0 do 1 z przykładów, na które model odpowiedział "
    "(nieprawidłowe wyjście liczy się jako 0). Ocenę tworzą sprawdzenia "
    "deterministyczne (kwoty, daty, identyfikatory, werdykty) i, tam gdzie trzeba, "
    "sędzia LLM dla pól opisowych."
)
METHOD_COST = (
    f"- Koszt: USD za {CALLS_PER_COST_UNIT} wywołań, z tokenów wszystkich prób; "
    "modele na GB10 kosztują 0 (własny sprzęt, bez energii i amortyzacji)."
)

type Cell = Callable[[Summary], str]
type Grid = Mapping[tuple[str, str], Summary]


@dataclass(frozen=True)
class RunInfo:
    """Facts about a run that the report states."""

    day: date
    cases: Sequence[str]
    models: Mapping[str, str]
    """Route name to the model that answers it (for example ``basal``)."""
    judge: str | None
    examples: Mapping[str, int]
    """Case name to the number of golden examples."""
    prices: Mapping[str, str]
    """Route name to a readable price."""
    judge_cost: float | None = None


def _percent(value: float | None) -> str:
    return NO_DATA if value is None else f"{value * 100:.0f}%"


def _seconds(value: float | None) -> str:
    return NO_DATA if value is None else f"{value:.1f}"


def _usd(value: float | None) -> str:
    return NO_DATA if value is None else f"{value:.4f}"


def _accuracy(summary: Summary) -> str:
    return _percent(summary.accuracy)


def _latency(summary: Summary) -> str:
    if summary.p50_seconds is None or summary.p95_seconds is None:
        return NO_DATA
    return f"{_seconds(summary.p50_seconds)} / {_seconds(summary.p95_seconds)}"


def _cost(summary: Summary) -> str:
    return _usd(summary.cost_per_100)


def _failures(summary: Summary) -> str:
    return f"{_percent(summary.invalid_rate)} / {_percent(summary.error_rate)}"


def _languages(summary: Summary) -> str:
    return f"{_percent(summary.pl_accuracy)} / {_percent(summary.en_accuracy)}"


def _judge_line(info: RunInfo) -> str:
    judge = (
        f"`{info.judge}`" if info.judge else "brak (tylko sprawdzenia deterministyczne)"
    )
    cost = (
        f", koszt sędziego w tym przebiegu: {_usd(info.judge_cost)} USD"
        if info.judge_cost is not None
        else ""
    )
    return f"- Sędzia: {judge}{cost}."


def _grid(title: str, note: str, grid: Grid, info: RunInfo, cell: Cell) -> list[str]:
    header = ["Trasa", *info.cases]
    lines = [f"### {title}", "", note, "", "| " + " | ".join(header) + " |"]
    lines.append("| --- |" + " ---: |" * len(info.cases))
    for model in info.models:
        cells = [
            cell(grid[case, model]) if (case, model) in grid else NO_DATA
            for case in info.cases
        ]
        lines.append("| " + " | ".join([f"`{model}`", *cells]) + " |")
    return [*lines, ""]


def render(summaries: Sequence[Summary], info: RunInfo) -> str:
    """Render the report.

    Args:
        summaries: One summary per model and case that was run.
        info: Facts about the run.

    Returns:
        The Markdown text, ending with the empty recommendations section.
    """
    grid = {(s.case, s.model): s for s in summaries}
    lines = [
        f"# Benchmark modeli LLM {info.day.isoformat()}",
        "",
        INTRO,
        "",
        _judge_line(info),
        METHOD_ACCURACY,
        METHOD_COST,
        f"- Zaliczony przykład: ocena co najmniej {PASS_SCORE:.2f}.",
        "",
        "## Zestaw",
        "",
        "| Przypadek | Przykłady |",
        "| --- | ---: |",
        *(f"| `{case}` | {info.examples.get(case, 0)} |" for case in info.cases),
        "",
        "| Trasa katalogu | Model | Cena za 1 mln tokenów (wejście / wyjście) |",
        "| --- | --- | --- |",
        *(
            f"| `tuttitrip:{route}` | `{name}` | {info.prices.get(route, NO_DATA)} |"
            for route, name in info.models.items()
        ),
        "",
        "## Wyniki",
        "",
        *_grid("Trafność", "Średnia ocena (0 do 100%).", grid, info, _accuracy),
        *_grid(
            "Opóźnienie p50 / p95 (s)",
            "Czas całego wywołania przypadku, z ponowieniami.",
            grid,
            info,
            _latency,
        ),
        *_grid(
            f"Koszt na {CALLS_PER_COST_UNIT} wywołań (USD)",
            "`n/d` = brak ceny modelu albo model nie brał udziału.",
            grid,
            info,
            _cost,
        ),
        *_grid(
            "Błędne wyjście / błąd dostawcy",
            "Odsetek odpowiedzi bez poprawnego wyjścia strukturalnego i odsetek "
            "awarii dostawcy (sieć, limit, timeout).",
            grid,
            info,
            _failures,
        ),
        *_grid(
            "Trafność po polsku / po angielsku",
            "Średnia ocena osobno dla przykładów PL i EN.",
            grid,
            info,
            _languages,
        ),
        RECOMMENDATIONS_HEADING,
        "",
        "_Do uzupełnienia po przeglądzie wyników._",
        "",
    ]
    return "\n".join(lines)
