"""Transition-path tests for the hybrid package reorganization.

Old import paths remain re-export-only shims for teammate branches and archived
evaluation scripts.  The canonical implementations live exclusively in the
``method1`` and ``method3`` subpackages.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path

import pytest


MODULE_SHIMS = (
    (
        "senhance.pipeline.hybrid.alignment",
        "senhance.pipeline.hybrid.method3.alignment",
    ),
    (
        "senhance.pipeline.hybrid.band_blender",
        "senhance.pipeline.hybrid.method3.band_blender",
    ),
    (
        "senhance.pipeline.hybrid.bands",
        "senhance.pipeline.hybrid.method3.bands",
    ),
    (
        "senhance.pipeline.hybrid.blender",
        "senhance.pipeline.hybrid.method3.blender",
    ),
    (
        "senhance.pipeline.hybrid.config",
        "senhance.pipeline.hybrid.method3.config",
    ),
    (
        "senhance.pipeline.hybrid.method3_band_config",
        "senhance.pipeline.hybrid.method3.method3_band_config",
    ),
    (
        "senhance.pipeline.hybrid.method3_config",
        "senhance.pipeline.hybrid.method3.method3_config",
    ),
    (
        "senhance.pipeline.hybrid.paired_stft",
        "senhance.pipeline.hybrid.method3.paired_stft",
    ),
    (
        "senhance.pipeline.hybrid_method1.alignment",
        "senhance.pipeline.hybrid.method1.alignment",
    ),
    (
        "senhance.pipeline.hybrid_method1.config",
        "senhance.pipeline.hybrid.method1.config",
    ),
    (
        "senhance.pipeline.hybrid_method1.keep_map",
        "senhance.pipeline.hybrid.method1.keep_map",
    ),
    (
        "senhance.pipeline.hybrid_method1.paired_stft",
        "senhance.pipeline.hybrid.method1.paired_stft",
    ),
    (
        "senhance.pipeline.hybrid_method1.processor",
        "senhance.pipeline.hybrid.method1.processor",
    ),
    (
        "senhance.pipeline.hybrid_method1.safety_controller",
        "senhance.pipeline.hybrid.method1.safety_controller",
    ),
)


def _implementation_members(module) -> dict[str, object]:
    return {
        name: value
        for name, value in vars(module).items()
        if not name.startswith("_")
        and (inspect.isclass(value) or inspect.isfunction(value))
        and getattr(value, "__module__", None) == module.__name__
    }


@pytest.mark.parametrize(
    ("legacy_name", "canonical_name"),
    MODULE_SHIMS,
    ids=lambda value: value.rsplit(".", 1)[-1],
)
def test_legacy_module_reexports_canonical_objects_by_identity(
    legacy_name: str,
    canonical_name: str,
) -> None:
    legacy = importlib.import_module(legacy_name)
    canonical = importlib.import_module(canonical_name)
    public_implementations = _implementation_members(canonical)

    assert public_implementations
    assert all(
        getattr(legacy, name, None) is value
        for name, value in public_implementations.items()
    )


@pytest.mark.parametrize(
    "legacy_name",
    [legacy_name for legacy_name, _ in MODULE_SHIMS],
    ids=lambda value: value.rsplit(".", 1)[-1],
)
def test_legacy_module_contains_no_algorithm_definitions(legacy_name: str) -> None:
    spec = importlib.util.find_spec(legacy_name)
    assert spec is not None and spec.origin is not None
    path = Path(spec.origin)
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    definitions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert definitions == []


def test_legacy_method1_package_exports_are_canonical_objects() -> None:
    legacy = importlib.import_module("senhance.pipeline.hybrid_method1")
    canonical = importlib.import_module("senhance.pipeline.hybrid.method1")

    assert legacy.__all__ == canonical.__all__
    assert all(getattr(legacy, name) is getattr(canonical, name) for name in canonical.__all__)


def test_legacy_method3_package_exports_are_canonical_objects() -> None:
    legacy = importlib.import_module("senhance.pipeline.hybrid")
    canonical = importlib.import_module("senhance.pipeline.hybrid.method3")

    assert all(name in legacy.__all__ for name in canonical.__all__)
    assert all(getattr(legacy, name) is getattr(canonical, name) for name in canonical.__all__)
