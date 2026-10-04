"""Reading a dotenv-style key file (pure)."""

from tuttitrip_worker.bench.constants import ASSIGN


def parse_env(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines.

    Comments and blank lines are skipped, a leading ``export`` is ignored and
    one pair of surrounding quotes is removed. Values are returned only to the
    caller; nothing here logs them.

    Args:
        text: Content of an env file.

    Returns:
        The variables.
    """
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip().removeprefix("export ").strip()
        if not line or line.startswith("#") or ASSIGN not in line:
            continue
        key, _, value = line.partition(ASSIGN)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:  # ruff: ignore[magic-value-comparison]
            value = value[1:-1]
        values[key.strip()] = value
    return values
