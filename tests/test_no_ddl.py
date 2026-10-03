"""The backend owns the schema: worker code never issues DDL."""

import ast
import re
from pathlib import Path

import pytest

import tuttitrip_worker

PACKAGE_ROOT = Path(tuttitrip_worker.__file__).parent
SOURCES = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
FORBIDDEN_CALLS = {"create_all", "drop_all", "create", "drop", "run_migrations"}
DDL = re.compile(
    r"^\s*(CREATE|ALTER|DROP|TRUNCATE|GRANT|REVOKE)\s+"
    r"(TABLE|INDEX|SCHEMA|EXTENSION|ROLE|TYPE|VIEW|ON)\b",
    re.IGNORECASE,
)


def rel(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT.parent))


@pytest.mark.parametrize("path", SOURCES, ids=rel)
def test_no_ddl_calls_or_statements(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in FORBIDDEN_CALLS
        ):
            offenders.append(f"line {node.lineno}: .{node.func.attr}()")
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and DDL.search(node.value)
        ):
            offenders.append(f"line {node.lineno}: DDL string")
    assert offenders == []


def test_no_migration_tooling_is_installed_or_imported() -> None:
    imports = {
        alias.name.split(".")[0]
        for path in SOURCES
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for path in SOURCES
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ImportFrom)
    }
    assert "alembic" not in imports
