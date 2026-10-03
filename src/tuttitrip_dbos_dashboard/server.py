"""HTTP server (stdlib) for the read-only DBOS dashboard."""

import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import override
from urllib.parse import parse_qs, urlsplit

from tuttitrip_dbos_dashboard import render
from tuttitrip_dbos_dashboard.config import load_config
from tuttitrip_dbos_dashboard.source import STATUSES, DashboardSource, DbosSource

logger = logging.getLogger("tuttitrip_dbos_dashboard")
MAX_LIMIT = 500
DEFAULT_LIMIT = 100
# Set by tuttitrip-gateway from the oauth2-proxy session (display only).
USER_HEADER = "X-Auth-Request-Email"
HTML = "text/html; charset=utf-8"
CSP = "default-src 'none'; style-src 'unsafe-inline'"

type Response = tuple[HTTPStatus, str, str]


def _limit(raw: str | None) -> int:
    try:
        return min(max(int(raw or DEFAULT_LIMIT), 1), MAX_LIMIT)
    except ValueError:
        return DEFAULT_LIMIT


class Dashboard:
    """Routes requests to pages; independent of the HTTP server."""

    def __init__(self, source: DashboardSource) -> None:
        self.source = source

    def handle(self, path: str, user: str | None) -> Response:
        """Answer one GET request.

        Args:
            path: Request path with query string.
            user: Signed-in admin from the gateway header, if any.

        Returns:
            Status, body and content type.
        """
        url = urlsplit(path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        if url.path == "/healthz":
            return HTTPStatus.OK, "ok", "text/plain; charset=utf-8"
        try:
            if url.path == "/":
                return self.overview(query, user)
            if url.path == "/workflow":
                return self.workflow(query, user)
        except Exception:
            logger.exception("request failed: %s", url.path)
            message = "Błąd odczytu z bazy DBOS (szczegóły w logach)."
            return HTTPStatus.INTERNAL_SERVER_ERROR, render.not_found(message), HTML
        return HTTPStatus.NOT_FOUND, render.not_found(url.path), HTML

    def _env(self, query: dict[str, str]) -> str | None:
        envs = self.source.envs()
        env = query.get("env") or envs[0]
        return env if env in envs else None

    def overview(self, query: dict[str, str], user: str | None) -> Response:
        """Environment overview page.

        Args:
            query: Parsed query string (env, status, name, limit).
            user: Signed-in admin, if known.

        Returns:
            Status, body and content type.
        """
        env = self._env(query)
        if env is None:
            return HTTPStatus.NOT_FOUND, render.not_found("Nieznane środowisko."), HTML
        status = query.get("status") if query.get("status") in STATUSES else None
        name = (query.get("name") or "").strip() or None
        limit = _limit(query.get("limit"))
        workflows = self.source.workflows(env, status=status, name=name, limit=limit)
        filters = {"status": status or "", "name": name or "", "limit": str(limit)}
        page = render.overview(
            render.Overview(
                envs=self.source.envs(),
                env=env,
                workflows=workflows,
                queues=self.source.queues(env),
                filters=filters,
            ),
            user,
        )
        return HTTPStatus.OK, page, HTML

    def workflow(self, query: dict[str, str], user: str | None) -> Response:
        """Single workflow page.

        Args:
            query: Parsed query string (env, id).
            user: Signed-in admin, if known.

        Returns:
            Status, body and content type.
        """
        env = self._env(query)
        workflow_id = query.get("id", "")
        found = self.source.workflow(env, workflow_id) if env and workflow_id else None
        if env is None or found is None:
            return (
                HTTPStatus.NOT_FOUND,
                render.not_found("Nie ma takiego workflow."),
                HTML,
            )
        steps = self.source.steps(env, workflow_id)
        return HTTPStatus.OK, render.workflow_page(env, found, steps, user), HTML


def make_handler(source: DashboardSource) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to a data source.

    Args:
        source: Where workflows and queues come from.

    Returns:
        A handler class for ``ThreadingHTTPServer``.
    """
    dashboard = Dashboard(source)

    class Handler(BaseHTTPRequestHandler):
        server_version = "tuttitrip-dbos-dashboard"

        def do_GET(self) -> None:
            """Serve /, /workflow and /healthz."""
            user = self.headers.get(USER_HEADER) or None
            status, body, content_type = dashboard.handle(self.path, user)
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.end_headers()
            self.wfile.write(data)

        @override
        def log_message(self, format: str, *args: object) -> None:
            """Log requests through ``logging`` instead of stderr."""
            logger.info("%s %s", self.address_string(), format % args)

    return Handler


def main() -> None:
    """Run the dashboard until the process is stopped."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    config = load_config()
    source = DbosSource(config.databases)
    # Only reachable on the private tuttitrip-admin network (behind the gateway).
    address = ("0.0.0.0", config.port)  # ruff: ignore[hardcoded-bind-all-interfaces]
    server = ThreadingHTTPServer(address, make_handler(source))
    logger.info(
        "DBOS dashboard on :%d for %s", config.port, ", ".join(config.databases)
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        source.close()
