"""Fixed values of the planning workflows."""

from typing import Final

PROGRESS_DRAFTING: Final = ("drafting", 10)
"""``progress`` event while the agent drafts the plan: ``(stage, percent)``."""

PROGRESS_SAVING: Final = ("saving", 90)
"""``progress`` event while the plan is stored."""

PROGRESS_LOADING: Final = ("loading", 5)
"""``progress`` event while the verdicts of a plan version are read."""

PROGRESS_WRITING_FROM: Final = 10
"""Percent when the first batch of justifications is written."""

PROGRESS_WRITING_SPAN: Final = 80
"""Percent range the batches of ``write_justifications`` fill together."""

PROGRESS_WRITING_STAGE: Final = "writing"
"""Stage name of the ``progress`` event while the model writes the texts."""

JUSTIFICATION_RETRIES: Final = 2
"""Extra tries the model gets to fix numbers or names that are not in the data."""

JUSTIFICATION_TEMPERATURE: Final = 0.0
"""Sampling temperature of the justifier: as repeatable as the model allows."""

LANGUAGE_NAMES: Final = {"pl": "Polish", "en": "English"}
"""English name of each ``locale`` of the contract, for the prompt."""

MAX_SENTENCES: Final = 2
"""Most sentences of one justification."""

MAX_DECIMALS: Final = 2
"""Most decimals a number may be rounded to when restated."""

PERCENT: Final = 100
"""A fraction written as a percent is multiplied by this."""

MIN_NAME_STEM: Final = 2
"""Shortest start of a name that still identifies it."""

NAME_ENDING_CHARS: Final = 2
"""Letters cut from a name before comparing it with a declined form."""

NUMBER_TOLERANCE: Final = 1e-9
"""Two numbers closer than this are equal."""

NUMBER_PATTERN: Final = r"\d+(?:[.,]\d+)?"
"""A number in a text, with a decimal point or comma."""

WORD_PATTERN: Final = r"[^\W\d_]+"
"""A word of letters in a text."""

SENTENCE_END_PATTERN: Final = r"(?<=[.!?])\s+"
"""Where one sentence ends and the next begins."""

PLAN_VERSION_KIND: Final = "plan version"
"""Name of the missing record in the ``document_not_found`` error."""

MAX_JUSTIFICATION_CHARS: Final = 400
"""Longest justification (the contract allows 600; two sentences need far less)."""
