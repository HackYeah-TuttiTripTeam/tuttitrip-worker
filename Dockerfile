# syntax=docker/dockerfile:1
# Multi-stage build following the official uv Docker guide
# (https://docs.astral.sh/uv/guides/integration/docker/, "non-editable install"):
# uv and the source tree stay in the builder; the runtime image gets only the
# virtual environment with the package installed into it.

FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app
# Dependencies first (cached layer), then the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-editable
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-editable


# Must match the builder's interpreter path (/usr/local/bin/python3.14).
FROM python:3.14-slim-trixie

RUN groupadd --system --gid 999 nonroot \
 && useradd --system --gid 999 --uid 999 --create-home nonroot

COPY --from=builder --chown=nonroot:nonroot /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER nonroot
WORKDIR /home/nonroot

# The worker serves no HTTP. Its main loop touches a liveness file after a
# successful query of the DBOS system database; the check reads its age.
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-m", "tuttitrip_worker.healthcheck"]

# SIGTERM -> running workflows get TUTTITRIP_DBOS__SHUTDOWN_TIMEOUT_SEC to
# finish; the rest are recovered on the next start. Exec form: Python is PID 1
# and receives the signal. Use `docker stop -t 40` (deploy sets --stop-timeout).
STOPSIGNAL SIGTERM
CMD ["tuttitrip-worker"]
