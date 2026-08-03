"""Strict Hybrid Method 2 configuration tests."""

from __future__ import annotations

import copy
import dataclasses

import pytest
import yaml

from senhance.pipeline.hybrid.method2 import load_method2_config

from tests.unit.hybrid.method2.conftest import METHOD2_CONFIG_PATH


def _raw_config():
    return yaml.safe_load(METHOD2_CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path, raw):
    path = tmp_path / "method2.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_project_config_loads_expected_policy(method2_config):
    assert method2_config.mode == "dsp_guided_level_calibration"
    assert method2_config.sample_rate == 48_000
    assert method2_config.level.epsilon == 1.0e-12
    assert method2_config.level.minimum_gain == 1.0
    assert method2_config.level.maximum_gain == 1.03
    assert method2_config.level.silence_rms_threshold == 1.0e-7
    assert method2_config.safety.peak_limit == 0.999
    assert method2_config.safety.clipping_threshold == 1.0


def test_all_config_dataclasses_are_frozen(method2_config):
    for value in (method2_config, method2_config.level, method2_config.safety):
        field = dataclasses.fields(value)[0].name
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(value, field, getattr(value, field))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.update({"extra": 1}), "Unknown root"),
        (lambda raw: raw.pop("method_2"), "Missing root"),
        (lambda raw: raw["method_2"].update({"extra": 1}), "Unknown method_2"),
        (lambda raw: raw["method_2"].pop("level"), "Missing method_2"),
        (
            lambda raw: raw["method_2"]["level"].update({"extra": 1}),
            "Unknown method_2.level",
        ),
        (
            lambda raw: raw["method_2"]["safety"].pop("peak_limit"),
            "Missing method_2.safety",
        ),
    ],
)
def test_loader_rejects_missing_and_unknown_fields(tmp_path, mutation, message):
    raw = copy.deepcopy(_raw_config())
    mutation(raw)
    with pytest.raises(ValueError, match=message):
        load_method2_config(_write_config(tmp_path, raw))


@pytest.mark.parametrize(
    ("path", "value", "exception", "message"),
    [
        (("sample_rate",), True, TypeError, "sample_rate.*integer"),
        (("sample_rate",), 16_000, ValueError, "48000"),
        (("mode",), "waveform_blend", ValueError, "mode"),
        (("level", "epsilon"), 1, TypeError, "epsilon.*floating"),
        (("level", "epsilon"), 0.0, ValueError, "epsilon.*positive"),
        (("level", "minimum_gain"), 0.0, ValueError, "minimum_gain.*positive"),
        (("level", "maximum_gain"), 0.5, ValueError, "maximum_gain"),
        (("level", "silence_rms_threshold"), -1.0, ValueError, "silence"),
        (("safety", "peak_limit"), 1.1, ValueError, "peak_limit"),
        (("safety", "clipping_threshold"), 0.0, ValueError, "clipping"),
    ],
)
def test_loader_rejects_invalid_values(tmp_path, path, value, exception, message):
    raw = copy.deepcopy(_raw_config())
    target = raw["method_2"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(exception, match=message):
        load_method2_config(_write_config(tmp_path, raw))


def test_missing_config_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="Method 2 config"):
        load_method2_config(tmp_path / "missing.yaml")
