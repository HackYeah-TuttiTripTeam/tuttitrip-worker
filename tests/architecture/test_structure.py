"""Structural rules: every domain slice has the same, predictable layout."""

from pathlib import Path

import pytest

from tests.architecture.layout import (
    DOMAIN_FILES,
    LAYER_PACKAGES,
    PACKAGE_ROOT,
    REQUIRED_DOMAIN_FILES,
    SHARED,
    SHARED_SUBPACKAGES,
    TOP_LEVEL_FILES,
    all_domains,
    all_modules,
    imports_in,
    is_package,
    rel,
    subdomains,
    subpackages,
)

DOMAINS = all_domains()


def test_there_are_domains_to_check() -> None:
    assert DOMAINS


def test_top_level_holds_only_domains_and_known_modules() -> None:
    stray = [
        p.name
        for p in PACKAGE_ROOT.iterdir()
        if p.name != "__pycache__"
        and not (p.is_file() and p.name in TOP_LEVEL_FILES)
        and not is_package(p)
    ]
    assert stray == []


def test_shared_has_the_documented_subpackages() -> None:
    names = {p.name for p in subpackages(PACKAGE_ROOT / SHARED)}
    assert names == SHARED_SUBPACKAGES


@pytest.mark.parametrize("domain", DOMAINS, ids=rel)
def test_domain_has_required_files(domain: Path) -> None:
    missing = [f for f in REQUIRED_DOMAIN_FILES if not (domain / f).is_file()]
    assert missing == []


@pytest.mark.parametrize("domain", DOMAINS, ids=rel)
def test_domain_contains_only_known_files(domain: Path) -> None:
    sub_names = {p.name for p in subdomains(domain)}
    stray = [
        p.name
        for p in domain.iterdir()
        if p.name != "__pycache__"
        and p.name not in DOMAIN_FILES
        and p.name not in LAYER_PACKAGES
        and p.name not in sub_names
    ]
    assert stray == [], "put code in steps.py/agents.py/logic/services, or a subdomain"


@pytest.mark.parametrize("domain", DOMAINS, ids=rel)
def test_layer_packages_have_at_least_one_module(domain: Path) -> None:
    for layer in LAYER_PACKAGES:
        package = domain / layer
        if not package.exists():
            continue
        assert is_package(package), f"{rel(package)} needs an __init__.py"
        assert [p for p in package.glob("*.py") if p.name != "__init__.py"]


@pytest.mark.parametrize(
    "init", [p for p in all_modules() if p.name == "__init__.py"], ids=rel
)
def test_init_modules_import_nothing(init: Path) -> None:
    # pytest-archon reads import statements only; it does not model that
    # importing `a.b.c` executes `a/__init__.py`. Import-free `__init__.py`
    # files keep its import graph equal to what runs.
    assert imports_in(init) == []
