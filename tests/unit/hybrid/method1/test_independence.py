"""Architecture guards proving Method 1 is independent from existing methods."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import yaml

import senhance.pipeline.hybrid.method1 as method1


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
METHOD1_SOURCE = SOURCE_ROOT / "senhance" / "pipeline" / "hybrid" / "method1"
METHOD1_CONFIG = PROJECT_ROOT / "config" / "hybrid_method_1.yaml"

FORBIDDEN_EXACT_OR_PREFIXES = (
    "senhance.config.settings",
    "senhance.pipeline.dl",
    "senhance.pipeline.dsp",
    "senhance.pipeline.improved_dsp",
    "senhance.pipeline.hybrid.method3",
)


def _source_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_path = path.relative_to(SOURCE_ROOT).with_suffix("")
    package = ".".join(relative_path.parts[:-1])
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative_name = "." * node.level + (node.module or "")
                imports.add(importlib.util.resolve_name(relative_name, package))
            elif node.module is not None:
                imports.add(node.module)
    return imports


def _forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(prefix + ".")
        for prefix in FORBIDDEN_EXACT_OR_PREFIXES
    )


def _mapping_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key).lower())
            keys.update(_mapping_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(_mapping_keys(child))
    return keys


def test_method1_source_imports_only_its_own_project_package():
    violations: list[str] = []
    for path in sorted(METHOD1_SOURCE.rglob("*.py")):
        for imported in sorted(_source_imports(path)):
            if _forbidden(imported):
                violations.append(f"{path.name}: {imported}")
            if imported.startswith("senhance.") and not imported.startswith(
                "senhance.pipeline.hybrid.method1"
            ):
                violations.append(f"{path.name}: external project import {imported}")
    assert violations == []


def test_method1_is_a_separate_sibling_package_not_a_method3_submodule():
    expected_files = {
        "__init__.py",
        "alignment.py",
        "config.py",
        "keep_map.py",
        "paired_stft.py",
        "processor.py",
        "safety_controller.py",
    }
    assert {path.name for path in METHOD1_SOURCE.glob("*.py")} == expected_files
    assert METHOD1_SOURCE.parent.name == "hybrid"
    assert METHOD1_SOURCE.name == "method1"
    assert (METHOD1_SOURCE.parent / "method3").is_dir()


def test_method1_configuration_contains_no_complete_dsp_selector():
    raw = yaml.safe_load(METHOD1_CONFIG.read_text(encoding="utf-8"))
    assert set(raw) == {"method_1"}
    keys = _mapping_keys(raw)
    assert keys.isdisjoint(
        {
            "dsp",
            "selected_dsp",
            "selected_backend",
            "improved_dsp",
            "legacy_dsp",
            "method_3",
        }
    )


def test_method1_source_never_names_or_constructs_existing_enhancer_classes():
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(METHOD1_SOURCE.glob("*.py"))
    )
    for forbidden_name in (
        "DeepFilterNetPipeline",
        "DSPPipeline",
        "ImprovedDSPPipeline",
        "AppSettings",
        "EnhancementStrategy",
        "build_dsp_registry",
    ):
        assert forbidden_name not in combined


def test_public_exports_are_explicit_and_resolve():
    assert isinstance(method1.__all__, list)
    assert len(method1.__all__) == len(set(method1.__all__))
    assert all(hasattr(method1, name) for name in method1.__all__)
    assert {
        "Method1SafetyLayer",
        "estimate_raw_keep_map",
        "SafetyGainController",
        "Method1PairedSTFTCore",
        "FixedNoisyDLAligner",
        "load_method1_config",
    }.issubset(method1.__all__)
