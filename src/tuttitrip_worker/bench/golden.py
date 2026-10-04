"""Golden sets on disk: ``tests/golden/<case>/examples.jsonl`` and the rubrics."""

from pathlib import Path

from tuttitrip_worker.bench.constants import EXAMPLES_FILE, RUBRIC_FILE
from tuttitrip_worker.bench.schemas import Example


def load_examples(golden_dir: Path, case: str) -> list[Example]:
    """Read the examples of a case.

    Args:
        golden_dir: ``tests/golden``.
        case: Directory name of the case.

    Returns:
        The examples in file order.

    Raises:
        ValueError: An id occurs twice.
    """
    path = golden_dir / case / EXAMPLES_FILE
    examples = [
        Example.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [example.id for example in examples]
    if len(set(ids)) != len(ids):
        msg = f"{path}: example ids must be unique"
        raise ValueError(msg)
    return examples


def load_rubric(golden_dir: Path, case: str) -> str:
    """The common rubric followed by the one of the case, when it has its own.

    Args:
        golden_dir: ``tests/golden``.
        case: Directory name of the case.

    Returns:
        The rubric text the judge reads.
    """
    parts = [(golden_dir / RUBRIC_FILE).read_text(encoding="utf-8")]
    own = golden_dir / case / RUBRIC_FILE
    if own.is_file():
        parts.append(own.read_text(encoding="utf-8"))
    return "\n\n".join(parts)
