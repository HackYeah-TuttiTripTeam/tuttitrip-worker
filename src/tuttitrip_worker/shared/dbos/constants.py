"""Fixed values shared by every workflow."""

from typing import Final

PROGRESS_DONE: Final = ("done", 100)
"""Last ``progress`` event of a workflow: ``(stage, percent)``."""
