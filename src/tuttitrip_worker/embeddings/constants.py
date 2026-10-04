"""Fixed values of ``embed_texts``."""

from typing import Final

PROGRESS_EMBEDDING: Final = ("embedding", 10)
"""``progress`` event while the model embeds the texts: ``(stage, percent)``."""

PROGRESS_STORING: Final = ("storing", 70)
"""``progress`` event while the vectors are written."""
