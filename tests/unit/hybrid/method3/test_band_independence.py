"""Independence guards specific to Method 3 fixed-frequency-band blending."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
HYBRID_SOURCE = SOURCE_ROOT / "senhance" / "pipeline" / "hybrid" / "method3"
BAND_CONFIG = PROJECT_ROOT / "config" / "hybrid_method_3_bands.yaml"
BAND_MODULES = (
    HYBRID_SOURCE / "method3_band_config.py",
    HYBRID_SOURCE / "bands.py",
    HYBRID_SOURCE / "band_blender.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "senhance.config",
    "senhance.settings",
    "senhance.pipeline.dl",
    "senhance.pipeline.dsp",
    "senhance.pipeline.improved_dsp",
)
FORBIDDEN_SELECTOR_KEYS = {
    "dsp",
    "dsp_method",
    "selected_dsp",
    "selected_dsp_method",
    "baseline_dsp",
    "baseline_method",
}


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


def test_band_configuration_contains_only_hybrid_policy_not_a_dsp_selector() -> None:
    raw = yaml.safe_load(BAND_CONFIG.read_text(encoding="utf-8"))

    assert set(raw) == {"method_3_bands"}
    assert _mapping_keys(raw).isdisjoint(FORBIDDEN_SELECTOR_KEYS)


def test_band_modules_do_not_import_settings_or_existing_methods() -> None:
    violations: list[str] = []
    for path in BAND_MODULES:
        for imported_module in sorted(_source_imports(path)):
            if imported_module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.name}: {imported_module}")
            if imported_module == "settings" or imported_module.endswith(".settings"):
                violations.append(f"{path.name}: {imported_module}")

    assert violations == []


def test_band_modules_are_located_entirely_inside_the_hybrid_package() -> None:
    assert all(path.is_file() for path in BAND_MODULES)
    assert all(path.parent == HYBRID_SOURCE for path in BAND_MODULES)
