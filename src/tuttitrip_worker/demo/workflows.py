"""Scheduled workflow: reset the jury's demo account once a day."""

import logging
from datetime import datetime

from dbos import DBOS

from tuttitrip_worker.contracts import APPLICATION_NAME
from tuttitrip_worker.demo import steps
from tuttitrip_worker.demo.schemas import DemoResetResult
from tuttitrip_worker.demo.services.backend_client import DemoResetError
from tuttitrip_worker.shared.config.settings import get_settings

logger = logging.getLogger(APPLICATION_NAME)


@DBOS.workflow(name="reset_demo_account")
async def reset_demo_account(scheduled_at: datetime, context: object) -> None:
    """Scheduled daily (``SCHEDULED_WORKFLOWS``): restore the sample trips.

    Without a configured secret it only warns in ``local`` and fails the run
    elsewhere, so a missing secret never passes silently. When the backend
    reports the demo as switched off nothing happens, and that is not an error.
    A backend that refuses the secret (4xx) fails the run too.

    Args:
        scheduled_at: When the schedule fired (unused).
        context: Schedule context (unused).
    """
    del scheduled_at, context
    settings = get_settings()
    if not settings.demo.reset_secret.get_secret_value():
        logger.warning("demo reset skipped: no TUTTITRIP_DEMO__RESET_SECRET")
        if settings.environment != "local":
            msg = "TUTTITRIP_DEMO__RESET_SECRET is not set"
            raise DemoResetError(msg)
        return
    result = DemoResetResult.model_validate(await steps.post_demo_reset())
    if result.status == "refused":
        msg = "the backend refused the demo reset (wrong secret or no endpoint)"
        raise DemoResetError(msg)
    logger.info("demo reset: %s (%d trips)", result.status, result.trips)
