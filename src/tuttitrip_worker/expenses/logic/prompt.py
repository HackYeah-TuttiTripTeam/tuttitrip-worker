"""Prompt framing for untrusted text (pure)."""

from hashlib import sha256


def data_tag(text: str, seed: str) -> str:
    """Tag name that delimits the user's text and does not occur in it.

    The name is a hash of the seed and the text, so the author cannot know it
    in advance and cannot close the block early. It is deterministic, which a
    DBOS workflow replay needs.

    Args:
        text: The untrusted text.
        seed: Stable per-job value, such as the trip id.

    Returns:
        A tag name such as ``expense_1a2b3c4d5e6f7a8b``.
    """
    digest = sha256(f"{seed}\n{text}".encode()).hexdigest()
    tag = f"expense_{digest[:16]}"
    while tag in text:  # astronomically unlikely, but then the text wins
        digest = sha256(digest.encode()).hexdigest()
        tag = f"expense_{digest[:16]}"
    return tag


def frame_expense_text(text: str, seed: str, locale: str) -> str:
    """User prompt: the locale outside, the text as data inside a unique block.

    Args:
        text: The untrusted sentence.
        seed: Stable per-job value, such as the trip id.
        locale: Validated ``pl`` or ``en``.

    Returns:
        The prompt.
    """
    tag = data_tag(text, seed)
    return (
        f"Language of the user: {locale}\n"
        f"The typed expense is the data between <{tag}> and </{tag}>.\n"
        f"<{tag}>\n{text}\n</{tag}>"
    )
