"""Integrity checks for the reviewed five-method package organization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = PROJECT_ROOT / "docs/method_layout_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_canonical_hybrid_files_match_the_reviewed_layout_manifest() -> None:
    manifest = _manifest()

    assert manifest["algorithm_change"] is False
    methods = manifest["methods"]
    assert set(methods) == {"hybrid_method_1", "hybrid_method_3"}

    for method in methods.values():
        for relative_path, expected_digest in method["canonical_files"].items():
            path = PROJECT_ROOT / relative_path
            assert path.is_file(), relative_path
            assert _sha256(path) == expected_digest, relative_path


def test_moved_hybrid_algorithms_differ_only_by_the_import_prefix() -> None:
    methods = _manifest()["methods"]

    for method in methods.values():
        canonical_package = method["canonical_package"]
        former_package = method["former_package"]
        canonical_by_name = {
            Path(relative_path).name: PROJECT_ROOT / relative_path
            for relative_path in method["canonical_files"]
        }
        for file_name, expected_digest in method[
            "pre_migration_normalized_files"
        ].items():
            source = canonical_by_name[file_name].read_text(encoding="utf-8")
            normalized = source.replace(canonical_package, former_package)
            assert hashlib.sha256(normalized.encode()).hexdigest() == expected_digest


def test_standalone_method_sources_match_the_preserved_aggregate() -> None:
    manifest = _manifest()
    source_paths: list[Path] = []
    for relative_dir in (
        "src/senhance/pipeline/dsp",
        "src/senhance/pipeline/improved_dsp",
        "src/senhance/pipeline/dl",
    ):
        source_paths.extend((PROJECT_ROOT / relative_dir).rglob("*.py"))

    aggregate = hashlib.sha256()
    for path in sorted(source_paths, key=lambda item: item.as_posix()):
        digest = _sha256(path)
        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        aggregate.update(f"{digest}  {relative_path}\n".encode())

    assert len(source_paths) == manifest["standalone_source_file_count"]
    assert aggregate.hexdigest() == manifest["standalone_source_aggregate_sha256"]
