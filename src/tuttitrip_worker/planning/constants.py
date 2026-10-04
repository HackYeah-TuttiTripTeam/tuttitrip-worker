"""Fixed values of ``generate_trip_plan``."""

from typing import Final

PROGRESS_DRAFTING: Final = ("drafting", 10)
"""``progress`` event while the agent drafts the plan: ``(stage, percent)``."""

PROGRESS_SAVING: Final = ("saving", 90)
"""``progress`` event while the plan is stored."""
