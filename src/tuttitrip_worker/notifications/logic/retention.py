"""How long notifications live (pure)."""

from datetime import datetime, timedelta
from typing import Final, NamedTuple

READ_DAYS: Final = 90
"""A read notification is deleted after this many days."""
ANY_DAYS: Final = 180
"""Every notification, read or not, is deleted after this many days."""
BATCH_SIZE: Final = 1000
"""Rows per ``DELETE``: short transactions, the API's ``LISTEN`` waits on them."""
MAX_BATCHES: Final = 1000
"""Safety stop of one run (a million rows); the next day continues."""


class Cutoffs(NamedTuple):
    """Rows created before these moments are deleted."""

    read: datetime
    any: datetime


def cutoffs(now: datetime) -> Cutoffs:
    """Compute the retention cutoffs.

    Args:
        now: Moment of the run (the schedule time, so a replay is the same).

    Returns:
        Created-before moments for read rows and for all rows.
    """
    return Cutoffs(now - timedelta(days=READ_DAYS), now - timedelta(days=ANY_DAYS))
