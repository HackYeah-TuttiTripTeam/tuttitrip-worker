"""Where the backend's internal API is reachable from the worker."""

from typing import Final

RESET_PATH: Final = "/api/v1/internal/demo/reset"
API_PORT: Final = 8000
MAIN_ENV: Final = "main"
LOCAL_ENV: Final = "local"


def api_base_url(environment: str, override: str = "") -> str:
    """Base URL of the environment's API container on the Docker network.

    Container names follow ``deploy/CONVENTIONS.md`` of the backend:
    ``tuttitrip-api`` for ``main``, ``tuttitrip-api-<env>`` otherwise. The
    worker talks to the container itself, never through the public gateway.

    Args:
        environment: The worker's environment name (``TUTTITRIP_ENVIRONMENT``).
        override: Explicit base URL; wins when not empty.

    Returns:
        The URL without a trailing slash.
    """
    if override:
        return override.rstrip("/")
    if environment == LOCAL_ENV:
        return f"http://localhost:{API_PORT}"
    name = (
        "tuttitrip-api" if environment == MAIN_ENV else f"tuttitrip-api-{environment}"
    )
    return f"http://{name}:{API_PORT}"
