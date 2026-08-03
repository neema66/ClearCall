from __future__ import annotations

from pathlib import Path

import pytest

from senhance.pipeline.hybrid.method2 import load_method2_config


PROJECT_ROOT = Path(__file__).resolve().parents[4]
METHOD2_CONFIG_PATH = PROJECT_ROOT / "config" / "hybrid_method_2.yaml"


@pytest.fixture
def method2_config():
    return load_method2_config(METHOD2_CONFIG_PATH)
