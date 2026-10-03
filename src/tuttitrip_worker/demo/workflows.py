"""Scheduled workflow: reset the jury's demo account once a day."""

import logging
from datetime import datetime

from dbos import DBOS

from tuttitrip_worker.contracts import APPLICATION_NAME
from tuttitrip_worker.demo import steps
from tuttitrip_worker.demo.schemas import DemoResetResult
from tuttitrip_worker.shared.config.settings import get_settings

logger = logging.getLogger(APPLICATION_NAME)


@DBOS.workflow(name="reset_demo_account")
async def reset_demo_account(scheduled_at: datetime, context: object) -> None:
    """Scheduled daily (``SCHEDULED_WORKFLOWS``): restore the sample trips.

    Does nothing without a configured secret. When the backend reports the
    demo as switched off it does nothing either, and that is not an error.

    Args:
        scheduled_at: When the schedule fired (unused).
        context: Schedule context (unused).
    """
    del scheduled_at, context
    if not get_settings().demo.reset_secret.get_secret_value():
        logger.info("demo reset skipped: no TUTTITRIP_DEMO__RESET_SECRET")
        return
    result = DemoResetResult.model_validate(await steps.reset_demo_account())
    logger.info("demo reset: %s (%d trips)", result.status, result.trips)
