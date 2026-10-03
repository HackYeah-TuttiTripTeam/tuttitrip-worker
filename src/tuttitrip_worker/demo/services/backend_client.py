"""The call that asks the backend to reset the demo account."""

import httpx

from tuttitrip_worker.demo.schemas import DemoResetResult

TIMEOUT_SECONDS = 150.0  # the backend logs in to Auth0 and recreates four trips
USER_AGENT = "TuttiTripWorker/1.0 (+https://tuttitrip.gburek.app)"


class DemoResetError(Exception):
    """The backend refused or could not be reached; carries no secret."""


def build_client() -> httpx.AsyncClient:
    """Create the HTTP client for the internal call.

    Returns:
        A client with the worker's user agent and a timeout.
    """
    return httpx.AsyncClient(
        timeout=TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
    )


async def request_reset(
    client: httpx.AsyncClient, url: str, secret: str
) -> DemoResetResult:
    """POST the reset request with the shared secret.

    Args:
        client: HTTP client (see ``build_client``).
        url: Full URL of the internal reset endpoint.
        secret: Shared secret, sent as a bearer token; never logged.

    Returns:
        What the backend did.

    Raises:
        DemoResetError: Transport failure, a non-200 status or a bad body.
    """
    try:
        response = await client.post(url, headers={"Authorization": f"Bearer {secret}"})
    except httpx.HTTPError as exc:
        msg = f"demo reset request failed: {type(exc).__name__}"
        raise DemoResetError(msg) from None
    if response.status_code != httpx.codes.OK:
        msg = f"demo reset refused: HTTP {response.status_code}"
        raise DemoResetError(msg)
    try:
        return DemoResetResult.model_validate_json(response.content)
    except ValueError:
        msg = "demo reset answered with an unexpected body"
        raise DemoResetError(msg) from None
