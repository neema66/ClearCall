"""Repository-level guards for independent enhancement method ownership.

These tests inspect imports without importing optional runtime dependencies. Each
method may import its own package plus the shared pipeline contract,
configuration, and logging infrastructure; it may not import another method.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
PIPELINE_ROOT = SOURCE_ROOT / "senhance" / "pipeline"


@dataclass(frozen=True)
class MethodBoundary:
    name: str
    package: str
    source: Path


METHODS = (
    MethodBoundary("Original DSP", "senhance.pipeline.dsp", PIPELINE_ROOT / "dsp"),
    MethodBoundary(
        "Improved DSP",
        "senhance.pipeline.improved_dsp",
        PIPELINE_ROOT / "improved_dsp",
    ),
    MethodBoundary(
        "DeepFilterNet3",
        "senhance.pipeline.dl",
        PIPELINE_ROOT / "dl",
    ),
    MethodBoundary(
        "Hybrid Method 1",
        "senhance.pipeline.hybrid.method1",
        PIPELINE_ROOT / "hybrid" / "method1",
    ),
    MethodBoundary(
        "Hybrid Method 2",
        "senhance.pipeline.hybrid.method2",
        PIPELINE_ROOT / "hybrid" / "method2",
    ),
    MethodBoundary(
        "Hybrid Method 3",
        "senhance.pipeline.hybrid.method3",
        PIPELINE_ROOT / "hybrid" / "method3",
    ),
)

ALLOWED_SHARED_PROJECT_IMPORTS = (
    "senhance.pipeline.base",
    "senhance.config",
    "senhance.logging_setup",
)


def _matches_package(module: str, package: str) -> bool:
    return module == package or module.startswith(package + ".")


def _source_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_path = path.relative_to(SOURCE_ROOT).with_suffix("")
    containing_package = ".".join(relative_path.parts[:-1])
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                imports.add(importlib.util.resolve_name(relative_name, containing_package))
            elif node.module is not None:
                imports.add(node.module)
    return imports


def _project_import_is_allowed(module: str, boundary: MethodBoundary) -> bool:
    if not module.startswith("senhance."):
        return True
    if _matches_package(module, boundary.package):
        return True
    return any(_matches_package(module, shared) for shared in ALLOWED_SHARED_PROJECT_IMPORTS)


def test_all_six_method_implementations_have_distinct_source_directories() -> None:
    assert len({method.package for method in METHODS}) == len(METHODS)
    assert len({method.source.resolve() for method in METHODS}) == len(METHODS)
    assert all(method.source.is_dir() for method in METHODS)
    assert all(any(method.source.glob("*.py")) for method in METHODS)


@pytest.mark.parametrize("boundary", METHODS, ids=lambda item: item.name)
def test_method_imports_only_itself_or_shared_infrastructure(
    boundary: MethodBoundary,
) -> None:
    violations: list[str] = []
    for path in sorted(boundary.source.glob("*.py")):
        for imported in sorted(_source_imports(path)):
            if not _project_import_is_allowed(imported, boundary):
                relative_path = path.relative_to(PROJECT_ROOT)
                violations.append(f"{relative_path}: imports {imported}")

    assert violations == [], (
        f"{boundary.name} crossed a method boundary. "
        "Move method selection/composition into orchestration code: "
        + "; ".join(violations)
    )


@pytest.mark.parametrize("boundary", METHODS, ids=lambda item: item.name)
def test_method_does_not_name_another_method_package(boundary: MethodBoundary) -> None:
    other_packages = {
        method.package for method in METHODS if method.package != boundary.package
    }
    violations: list[str] = []
    for path in sorted(boundary.source.glob("*.py")):
        for imported in sorted(_source_imports(path)):
            if any(_matches_package(imported, package) for package in other_packages):
                violations.append(f"{path.name}: {imported}")

    assert violations == []
