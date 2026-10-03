"""HTML for the dashboard. Pure: everything shown is escaped here."""

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from urllib.parse import urlencode

from tuttitrip_dbos_dashboard.source import STATUSES, QueueRow, StepRow, WorkflowRow

_CSS = """
:root{color-scheme:light dark;--bg:#fff;--fg:#1d1d1f;--muted:#6e6e73;--line:#e3e3e8;
--ok:#1a7f37;--err:#c62828;--warn:#9a6700;--run:#0b63c5}
@media (prefers-color-scheme:dark){:root{--bg:#121214;--fg:#ececf1;--muted:#a1a1aa;
--line:#2c2c31;--ok:#4ac26b;--err:#ff7b72;--warn:#d29922;--run:#58a6ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}
a{color:var(--run)}.muted{color:var(--muted)}
nav a{margin-right:12px}nav a.on{font-weight:700;text-decoration:none;color:var(--fg)}
.table{overflow-x:auto}table{border-collapse:collapse;width:100%}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);
vertical-align:top;white-space:nowrap}td.wrap{white-space:pre-wrap;word-break:break-word}
.s-SUCCESS{color:var(--ok)}.s-ERROR,.s-MAX_RECOVERY_ATTEMPTS_EXCEEDED{color:var(--err)}
.s-CANCELLED{color:var(--warn)}.s-PENDING,.s-ENQUEUED{color:var(--run)}
form{display:flex;gap:8px;flex-wrap:wrap;align-items:end}
label{display:flex;flex-direction:column;font-size:12px;color:var(--muted)}
"""


def _time(epoch_ms: int | None) -> str:
    if epoch_ms is None:
        return "-"
    return datetime.fromtimestamp(epoch_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _duration(start_ms: int | None, end_ms: int | None) -> str:
    if start_ms is None or end_ms is None:
        return "-"
    return f"{(end_ms - start_ms) / 1000:.1f} s"


def _text(value: object) -> str:
    return "-" if value is None or not str(value) else escape(str(value))


def _status(status: str) -> str:
    return f'<span class="s-{escape(status)}">{escape(status)}</span>'


def _page(title: str, user: str | None, body: str) -> str:
    who = f'<p class="muted">Zalogowano: {escape(user)}</p>' if user else ""
    return (
        '<!doctype html><html lang="pl"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><main><h1>{escape(title)}</h1>{who}{body}</main></body></html>"
    )


def _env_nav(envs: list[str], current: str) -> str:
    links = "".join(
        f'<a href="/?{urlencode({"env": e})}"{" class=on" if e == current else ""}>'
        f"{escape(e)}</a>"
        for e in envs
    )
    return f"<nav>Środowisko: {links}</nav>"


@dataclass(frozen=True)
class Overview:
    """Data shown on the environment overview."""

    envs: list[str]
    env: str
    workflows: list[WorkflowRow]
    queues: list[QueueRow]
    filters: dict[str, str]


def overview(data: Overview, user: str | None) -> str:
    """Render the environment overview: queues, status counts, workflows.

    Args:
        data: Environments, workflows, queues and active filters.
        user: The signed-in admin, as passed by the gateway.

    Returns:
        A complete HTML page.
    """
    envs, env, workflows, queues, filters = (
        data.envs,
        data.env,
        data.workflows,
        data.queues,
        data.filters,
    )
    counts = Counter(w.status for w in workflows)
    summary = " · ".join(f"{_status(s)} {counts[s]}" for s in STATUSES if counts[s])
    queue_rows = "".join(
        f"<tr><td>{escape(q.name)}</td><td>{q.enqueued}</td><td>{q.pending}</td>"
        f"<td>{_text(q.concurrency)}</td><td>{_text(q.worker_concurrency)}</td></tr>"
        for q in queues
    )
    options = "".join(
        f"<option{' selected' if filters.get('status') == s else ''}>{s}</option>"
        for s in STATUSES
    )
    form = (
        f'<form method="get"><input type="hidden" name="env" value="{escape(env)}">'
        f'<label>Status<select name="status"><option value="">(wszystkie)</option>'
        f"{options}</select></label>"
        f'<label>Workflow<input name="name" value="{escape(filters.get("name", ""))}">'
        f'</label><label>Limit<input name="limit" type="number" min="1" max="500" '
        f'value="{escape(filters.get("limit", "100"))}"></label>'
        "<button>Filtruj</button></form>"
    )
    wf_rows = "".join(
        "<tr>"
        f'<td><a href="/workflow?{urlencode({"env": env, "id": w.workflow_id})}">'
        f"{escape(w.workflow_id)}</a></td>"
        f"<td>{escape(w.name)}</td><td>{_status(w.status)}</td>"
        f"<td>{_text(w.queue_name)}</td><td>{_text(w.app_version)}</td>"
        f"<td>{_time(w.created_at_ms)}</td>"
        f"<td>{_duration(w.created_at_ms, w.updated_at_ms)}</td>"
        f"<td>{_text(w.recovery_attempts)}</td></tr>"
        for w in workflows
    )
    body = (
        f"{_env_nav(envs, env)}"
        "<h2>Kolejki</h2><div class=table><table><tr><th>Kolejka</th><th>ENQUEUED</th>"
        "<th>PENDING</th><th>Limit globalny</th><th>Limit na workera</th></tr>"
        f"{queue_rows or '<tr><td colspan=5 class=muted>brak</td></tr>'}</table></div>"
        f"<h2>Workflowy ({len(workflows)} najnowszych)</h2>"
        f"<p>{summary or '-'}</p>{form}"
        "<div class=table><table><tr><th>ID</th><th>Workflow</th><th>Status</th>"
        "<th>Kolejka</th><th>Wersja</th><th>Utworzony (UTC)</th><th>Czas</th>"
        "<th>Próby</th></tr>"
        f"{wf_rows or '<tr><td colspan=8 class=muted>brak</td></tr>'}"
        "</table></div>"
    )
    return _page(f"DBOS · {env}", user, body)


def workflow_page(
    env: str, workflow: WorkflowRow, steps: list[StepRow], user: str | None
) -> str:
    """Render one workflow with its error and steps.

    Args:
        env: Environment name.
        workflow: The workflow.
        steps: Its recorded steps.
        user: The signed-in admin, as passed by the gateway.

    Returns:
        A complete HTML page.
    """
    facts = [
        ("Status", _status(workflow.status)),
        ("Workflow", escape(workflow.name)),
        ("Kolejka", _text(workflow.queue_name)),
        ("Wersja aplikacji", _text(workflow.app_version)),
        ("Executor", _text(workflow.executor_id)),
        ("Utworzony (UTC)", _time(workflow.created_at_ms)),
        ("Zmieniony (UTC)", _time(workflow.updated_at_ms)),
        ("Próby odzyskania", _text(workflow.recovery_attempts)),
        ("Błąd", _text(workflow.error)),
    ]
    fact_rows = "".join(
        f"<tr><th>{k}</th><td class=wrap>{v}</td></tr>" for k, v in facts
    )
    step_rows = "".join(
        f"<tr><td>{s.function_id}</td><td>{escape(s.function_name)}</td>"
        f"<td>{_time(s.started_at_ms)}</td>"
        f"<td>{_duration(s.started_at_ms, s.completed_at_ms)}</td>"
        f"<td>{_text(s.child_workflow_id)}</td>"
        f"<td class=wrap>{_text(s.error)}</td></tr>"
        for s in steps
    )
    body = (
        f'<p><a href="/?{urlencode({"env": env})}">← {escape(env)}</a></p>'
        f"<p class=muted>{escape(workflow.workflow_id)}</p>"
        f"<div class=table><table>{fact_rows}</table></div>"
        "<h2>Kroki</h2><div class=table><table><tr><th>#</th><th>Funkcja</th>"
        "<th>Start (UTC)</th><th>Czas</th><th>Workflow potomny</th><th>Błąd</th></tr>"
        f"{step_rows or '<tr><td colspan=6 class=muted>brak</td></tr>'}</table></div>"
    )
    return _page(f"DBOS · {env} · {workflow.name}", user, body)


def not_found(message: str) -> str:
    """Render a small error page.

    Args:
        message: What was not found.

    Returns:
        A complete HTML page.
    """
    return _page(
        "Nie znaleziono", None, f'<p>{escape(message)}</p><p><a href="/">←</a></p>'
    )
