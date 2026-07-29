from __future__ import annotations

from pathlib import Path

import pytest

from senhance.pipeline.hybrid.method1.config import Method1Config, load_method1_config


PROJECT_ROOT = Path(__file__).resolve().parents[4]
METHOD1_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_method_1.yaml"


@pytest.fixture
def method1_config() -> Method1Config:
    return load_method1_config(METHOD1_CONFIG_PATH)
