"""Framing of untrusted text for prompts (pure, shared by the domains)."""

from hashlib import sha256


def data_tag(text: str, seed: str, prefix: str) -> str:
    """Tag name that delimits untrusted text and does not occur in it.

    The name is a hash of the seed and the text, so the author of the text
    cannot know it in advance and cannot close the block early. It is
    deterministic, which a DBOS workflow replay needs.

    Args:
        text: The untrusted text.
        seed: Stable per-job value, such as a document or trip id.
        prefix: Readable start of the tag (``pasted``, ``expense``...).

    Returns:
        A tag name such as ``expense_1a2b3c4d5e6f7a8b``.
    """
    digest = sha256(f"{seed}\n{text}".encode()).hexdigest()
    tag = f"{prefix}_{digest[:16]}"
    while tag in text:  # astronomically unlikely, but then the text wins
        digest = sha256(digest.encode()).hexdigest()
        tag = f"{prefix}_{digest[:16]}"
    return tag


def data_block(text: str, seed: str, prefix: str, intro: str) -> str:
    """The text as data inside a unique block, after an intro line.

    Args:
        text: The untrusted text.
        seed: Stable per-job value.
        prefix: Readable start of the tag.
        intro: What the block is, e.g. ``The typed expense``.

    Returns:
        ``<intro> is the data between <tag> and </tag>.`` and the block.
    """
    tag = data_tag(text, seed, prefix)
    return (
        f"{intro} is the data between <{tag}> and </{tag}>.\n<{tag}>\n{text}\n</{tag}>"
    )
