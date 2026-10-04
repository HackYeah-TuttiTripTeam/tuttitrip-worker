"""Prompt framing for the pasted plan (pure)."""

from tuttitrip_worker.prompts import data_block
from tuttitrip_worker.prompts import data_tag as _data_tag

PREFIX = "pasted"


def data_tag(text: str, seed: str) -> str:
    """Tag name that delimits the pasted text and does not occur in it.

    Args:
        text: The untrusted pasted text.
        seed: Stable per-job value, such as the document id.

    Returns:
        A tag name such as ``pasted_1a2b3c4d5e6f7a8b``.
    """
    return _data_tag(text, seed, PREFIX)


def frame_pasted_text(text: str, seed: str, city_slug: str) -> str:
    """User prompt: the city outside, the text as data inside a unique block.

    Args:
        text: The untrusted pasted text.
        seed: Stable per-job value, such as the document id.
        city_slug: Validated slug from the payload (``[a-z0-9-]`` only).

    Returns:
        The prompt.
    """
    block = data_block(text, seed, PREFIX, "The pasted plan")
    return f"City slug: {city_slug}\n{block}"
