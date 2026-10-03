"""Docker healthcheck: is the worker's liveness file fresh?

The main loop touches the file every ``LIVENESS_INTERVAL_SEC`` seconds after
a successful query of the DBOS system database. Run with
``python -m tuttitrip_worker.healthcheck``; exit code 0 means healthy.
"""

import sys
import tempfile
import time
from pathlib import Path
from typing import Final

LIVENESS_FILE: Final = Path(tempfile.gettempdir()) / "tuttitrip-worker.alive"
LIVENESS_INTERVAL_SEC: Final = 15.0
MAX_AGE_SEC: Final = 4 * LIVENESS_INTERVAL_SEC


def is_alive(path: Path = LIVENESS_FILE, now: float | None = None) -> bool:
    """Check the liveness file.

    Args:
        path: Liveness file.
        now: Current time (seconds since the epoch); defaults to now.

    Returns:
        True if the file was touched within ``MAX_AGE_SEC``.
    """
    try:
        age = (time.time() if now is None else now) - path.stat().st_mtime
    except FileNotFoundError:
        return False
    return age <= MAX_AGE_SEC


if __name__ == "__main__":
    sys.exit(0 if is_alive() else 1)
