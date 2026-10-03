"""Write ``contracts/jobs.schema.json`` from ``tuttitrip_worker.contracts``.

Usage: ``uv run python scripts/export_contracts.py`` (``--check`` only compares).
"""

import sys
from pathlib import Path

from tuttitrip_worker.contracts import contract_json

TARGET = Path(__file__).resolve().parents[1] / "contracts" / "jobs.schema.json"


def main() -> int:
    """Write (or with ``--check`` verify) the schema file.

    Returns:
        Process exit code.
    """
    content = contract_json()
    if "--check" in sys.argv[1:]:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != content:
            print(f"{TARGET} is out of date; run scripts/export_contracts.py")
            return 1
        print(f"{TARGET} is up to date")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(content, encoding="utf-8")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
