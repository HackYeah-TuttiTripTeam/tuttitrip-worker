"""Prompt framing for untrusted text (pure)."""

from hashlib import sha256


def data_tag(text: str, seed: str) -> str:
    """Tag name that delimits the pasted text and does not occur in it.

    The name is a hash of the seed and the text, so the author of the pasted
    text cannot know it in advance and cannot close the block early. It is
    deterministic, which a DBOS workflow replay needs.

    Args:
        text: The untrusted pasted text.
        seed: Stable per-job value, such as the document id.

    Returns:
        A tag name such as ``pasted_1a2b3c4d5e6f7a8b``.
    """
    digest = sha256(f"{seed}\n{text}".encode()).hexdigest()
    tag = f"pasted_{digest[:16]}"
    while tag in text:  # astronomically unlikely, but then the text wins
        digest = sha256(digest.encode()).hexdigest()
        tag = f"pasted_{digest[:16]}"
    return tag


def frame_pasted_text(text: str, seed: str, city_slug: str) -> str:
    """User prompt: the city outside, the text as data inside a unique block.

    Args:
        text: The untrusted pasted text.
        seed: Stable per-job value, such as the document id.
        city_slug: Validated slug from the payload (``[a-z0-9-]`` only).

    Returns:
        The prompt.
    """
    tag = data_tag(text, seed)
    return (
        f"City slug: {city_slug}\n"
        f"The pasted plan is the data between <{tag}> and </{tag}>.\n"
        f"<{tag}>\n{text}\n</{tag}>"
    )
