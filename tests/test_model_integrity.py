from pathlib import Path

import pytest

from model_integrity import missing_sd15_model_files, validate_sd15_model, REQUIRED_CONFIG_FILES, REQUIRED_WEIGHT_OPTIONS


def _write_nonempty(path: Path, content=b"test"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_incomplete_model_reports_missing_component_weights(tmp_path):
    _write_nonempty(tmp_path / "model_index.json")
    missing = missing_sd15_model_files(tmp_path)
    assert "text_encoder/<valid safetensors weights>" in missing
    assert "unet/<valid safetensors weights>" in missing
    assert "vae/<valid safetensors weights>" in missing


def test_complete_model_passes_validation(tmp_path):
    for relative_path in REQUIRED_CONFIG_FILES:
        _write_nonempty(tmp_path / relative_path)
    for candidates in REQUIRED_WEIGHT_OPTIONS.values():
        _write_nonempty(tmp_path / candidates[0], b"\0" * (10_000_001))
    assert validate_sd15_model(tmp_path) == tmp_path


def test_missing_model_directory_is_reported(tmp_path):
    missing_dir = tmp_path / "does-not-exist"
    assert missing_sd15_model_files(missing_dir) == ["<model directory>"]
    with pytest.raises(RuntimeError, match="model is incomplete"):
        validate_sd15_model(missing_dir)
