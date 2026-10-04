"""Fixed values of ``parse_pasted_plan``."""

from typing import Final

PROGRESS_LOADING: Final = ("loading", 5)
"""``progress`` event while the pasted plan is loaded: ``(stage, percent)``."""

PROGRESS_READING: Final = ("reading", 15)
"""``progress`` event while the model reads the plan."""

PROGRESS_MATCHING: Final = ("matching", 60)
"""``progress`` event while the items are matched to the catalog."""

PROGRESS_SAVING: Final = ("saving", 90)
"""``progress`` event while the result is stored."""


LINE_BREAKS: Final = ("\n", "\r")
"""Characters that make a quote span more than one line."""

QUOTE_PREVIEW_CHARS: Final = 80
"""Characters of a rejected quote shown to the model when it must fix it."""
