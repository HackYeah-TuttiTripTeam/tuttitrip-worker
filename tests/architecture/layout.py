"""Discovery of domains and modules in the ``tuttitrip_worker`` package.

Conventions (see AGENTS.md):

* A *domain* is a direct subpackage of ``tuttitrip_worker`` other than
  ``shared``. A *subdomain* is a subpackage of a domain that is not one of its
  layer packages (``logic``, ``services``); it follows the same layout.
* *Pure modules* are ``tuttitrip_worker.contracts``, every ``schemas.py`` and
  every module in a ``logic/`` package. They may import only the standard
  library, Pydantic and other pure modules.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import tuttitrip_worker

PACKAGE = "tuttitrip_worker"
PACKAGE_ROOT = Path(tuttitrip_worker.__file__).parent
SHARED = "shared"
LAYER_PACKAGES = frozenset({"logic", "services"})
DOMAIN_FILES = frozenset(
    {
        "__init__.py",
        "workflows.py",
        "schemas.py",
        "steps.py",
        "agents.py",
    }
)
REQUIRED_DOMAIN_FILES = ("__init__.py", "workflows.py", "schemas.py")
TOP_LEVEL_FILES = frozenset(
    {
        "__init__.py",
        "contracts.py",
        "quotes.py",
        "main.py",
        "healthcheck.py",
        "py.typed",
    }
)
SHARED_SUBPACKAGES = frozenset({"config", "dbos", "llm", "db"})


def is_package(path: Path) -> bool:
    return path.is_dir() and (path / "__init__.py").is_file()


def subpackages(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if is_package(p))


def subdomains(domain: Path) -> list[Path]:
    return [p for p in subpackages(domain) if p.name not in LAYER_PACKAGES]


def walk_domains(domain: Path) -> Iterator[Path]:
    yield domain
    for sub in subdomains(domain):
        yield from walk_domains(sub)


def top_level_domains() -> list[str]:
    return [p.name for p in subpackages(PACKAGE_ROOT) if p.name != SHARED]


def all_domains() -> list[Path]:
    """Every domain and subdomain, depth first."""
    return [
        d
        for top in subpackages(PACKAGE_ROOT)
        if top.name != SHARED
        for d in walk_domains(top)
    ]


def all_modules() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def imports_in(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append("." * node.level + (node.module or ""))
    return names


def rel(path: Path) -> str:
    return str(path.relative_to(PACKAGE_ROOT.parent))
