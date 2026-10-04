"""Fixed values of ``extract_offer_evidence``."""

from typing import Final

PROGRESS_READING: Final = ("reading", 5)
"""``progress`` event while the pasted offer is loaded: ``(stage, percent)``."""

PROGRESS_ASSESSING: Final = ("assessing", 15)
"""``progress`` event while the model assesses the offer."""

PROGRESS_SAVING: Final = ("saving", 95)
"""``progress`` event while the result is stored."""
