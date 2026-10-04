"""I/O step of the demo domain: ask the backend to reset the demo account."""

from typing import Any

from dbos import DBOS

from tuttitrip_worker.demo.logic.target import RESET_PATH, api_base_url
from tuttitrip_worker.demo.services import backend_client
from tuttitrip_worker.shared.config.settings import get_settings


@DBOS.step(
    retries_allowed=True,
    max_attempts=get_settings().demo.reset_max_attempts,
    interval_seconds=get_settings().demo.reset_retry_interval_seconds,
)
async def post_demo_reset() -> dict[str, Any]:
    """Call the backend's internal reset (idempotent and atomic there).

    ``demo.reset_max_attempts`` attempts in total (DBOS ``max_attempts``) for
    transport errors and 5xx.

    Returns:
        JSON object matching ``DemoResetResult``.
    """
    settings = get_settings()
    url = api_base_url(settings.environment, settings.demo.api_base_url) + RESET_PATH
    async with backend_client.build_client() as client:
        result = await backend_client.request_reset(
            client, url, settings.demo.reset_secret.get_secret_value()
        )
    output: dict[str, Any] = result.model_dump(mode="json")
    return output
