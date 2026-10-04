"""Dependency rules between shared, domains and frameworks (pytest-archon).

pytest-archon 0.0.7 semantics (as pinned in the backend repo):
``should_not_import`` is transitive by default and follows imports through
every module of the checked package; ``only_direct_imports=True`` limits it
to a module's own import statements. A rule that matches no module fails.

Purity rules are transitive (a pure module must not reach a framework via any
chain). Layering rules ("who may import X directly") are direct, because
legitimate chains such as workflows -> steps -> sqlalchemy exist.
"""

import pytest
from pytest_archon import archrule

from tests.architecture.layout import PACKAGE, top_level_domains

DOMAINS = top_level_domains()
IO_FRAMEWORKS = (
    "pydantic_ai",
    "pydantic_ai_harness",
    "stackone_defender",
    "dbos",
    "sqlalchemy",
    "psycopg",
    "pgvector",
    "openai",
    "httpx",
)
PURE = (
    rf"^{PACKAGE}\.(contracts|quotes|prompts)$",
    r"\.schemas$",
    r"\.logic(\.|$)",
)


def _tree(name: str) -> tuple[str, str]:
    return (name, f"{name}.*")


@pytest.mark.parametrize("domain", DOMAINS)
def test_shared_does_not_import_domains(domain: str) -> None:
    (
        archrule("shared kernel is independent", comment="shared must not know domains")
        .match(*_tree(f"{PACKAGE}.shared"))
        .should_not_import(*_tree(f"{PACKAGE}.{domain}"))
        .check(PACKAGE)
    )


@pytest.mark.parametrize("domain", DOMAINS)
def test_domains_do_not_import_each_others_internals(domain: str) -> None:
    others = [d for d in DOMAINS if d != domain]
    (
        archrule(
            "cross-domain access",
            comment="only another domain's schemas; share code through shared/",
        )
        .match(*_tree(f"{PACKAGE}.{domain}"))
        .should_not_import(
            *[p for other in others for p in _tree(f"{PACKAGE}.{other}")]
        )
        .may_import(*[f"{PACKAGE}.{other}.schemas" for other in others])
        .check(PACKAGE, only_direct_imports=True)
    )


@pytest.mark.parametrize("framework", IO_FRAMEWORKS)
def test_pure_modules_do_not_reach_io_frameworks(framework: str) -> None:
    (
        archrule("pure modules are framework-free", use_regex=True)
        .match(*PURE)
        .should_not_import(rf"^{framework}(\.|$)")
        .check(PACKAGE)
    )


def test_pure_modules_do_not_reach_io_layers() -> None:
    (
        archrule("pure modules stay pure", use_regex=True)
        .match(*PURE)
        .should_not_import(
            r"\.(workflows|steps|agents)$",
            r"\.services(\.|$)",
            rf"^{PACKAGE}\.shared(\.|$)",
            rf"^{PACKAGE}\.main$",
        )
        .check(PACKAGE)
    )


def test_only_agents_and_shared_llm_import_pydantic_ai() -> None:
    (
        archrule("agents and Harness live in agents.py", use_regex=True)
        .match(rf"^{PACKAGE}(\.|$)")
        .exclude(r"\.agents$", rf"^{PACKAGE}\.shared\.llm(\.|$)")
        .should_not_import(
            r"^(pydantic_ai|pydantic_ai_harness|stackone_defender)(\.|$)"
        )
        .check(PACKAGE, only_direct_imports=True)
    )


def test_only_runtime_layers_import_dbos() -> None:
    (
        archrule("DBOS stays in workflows/steps/runtime", use_regex=True)
        .match(rf"^{PACKAGE}(\.|$)")
        .exclude(
            r"\.(workflows|steps)$",
            rf"^{PACKAGE}\.shared\.(dbos|db)(\.|$)",
            rf"^{PACKAGE}\.main$",
        )
        .should_not_import(r"^dbos(\.|$)")
        .check(PACKAGE, only_direct_imports=True)
    )


def test_only_steps_and_shared_db_touch_the_database() -> None:
    (
        archrule("database I/O happens in steps", use_regex=True)
        .match(rf"^{PACKAGE}(\.|$)")
        .exclude(r"\.steps$", rf"^{PACKAGE}\.shared\.db(\.|$)")
        .should_not_import(r"^sqlalchemy(\.|$)", r"^psycopg(\.|$)", r"^pgvector(\.|$)")
        .check(PACKAGE, only_direct_imports=True)
    )
