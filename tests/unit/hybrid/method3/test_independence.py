"""Architecture guards for the standalone hybrid signal core."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[4]
HYBRID_SOURCE = PROJECT_ROOT / "src" / "senhance" / "pipeline" / "hybrid" / "method3"
HYBRID_CONFIG = PROJECT_ROOT / "config" / "hybrid.yaml"
METHOD3_CONFIG = PROJECT_ROOT / "config" / "hybrid_method_3.yaml"

FORBIDDEN_IMPORT_PREFIXES = (
    "senhance.config.settings",
    "senhance.pipeline.dl",
    "senhance.pipeline.dsp",
    "senhance.pipeline.improved_dsp",
    "senhance.pipeline.hybrid.method1",
)


def _source_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    relative_path = path.relative_to(PROJECT_ROOT / "src").with_suffix("")
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


def test_hybrid_source_does_not_import_existing_method_implementations() -> None:
    violations: list[str] = []
    for path in sorted(HYBRID_SOURCE.rglob("*.py")):
        for imported_module in sorted(_source_imports(path)):
            if imported_module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                violations.append(f"{path.name}: {imported_module}")

    assert violations == []


def test_hybrid_configuration_does_not_select_or_embed_a_dsp_method() -> None:
    raw = yaml.safe_load(HYBRID_CONFIG.read_text(encoding="utf-8"))

    assert set(raw) == {"hybrid"}
    assert "dsp" not in raw["hybrid"]
    assert "selected_dsp" not in HYBRID_CONFIG.read_text(encoding="utf-8")


def test_method3_configuration_selects_replaceable_dsp_without_code_dependency() -> None:
    raw = yaml.safe_load(METHOD3_CONFIG.read_text(encoding="utf-8"))

    assert set(raw) == {"method_3"}
    dsp = raw["method_3"]["dsp"]
    assert dsp["selected_backend"] == "improved_dsp"
    assert set(dsp["backends"]) == {"improved_dsp", "legacy_dsp"}
    assert all(
        set(profile) == {"dl_minus_dsp_delay_samples"} for profile in dsp["backends"].values()
    )


def test_waveform_blender_does_not_route_through_paired_stft() -> None:
    imports = _source_imports(HYBRID_SOURCE / "blender.py")

    assert "senhance.pipeline.hybrid.method3.paired_stft" not in imports
