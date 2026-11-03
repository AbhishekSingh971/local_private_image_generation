import json
import struct
from pathlib import Path

import pytest

from model_integrity import (
    REQUIRED_CONFIG_FILES,
    REQUIRED_WEIGHT_OPTIONS,
    missing_animatediff_adapter_files,
    missing_sd15_model_files,
    validate_animatediff_adapter,
    validate_sd15_model,
)


def _write_nonempty(path: Path, content=b"test"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_valid_safetensors(path: Path):
    # A tiny valid header plus a >10 MB payload exercises header validation
    # without allocating tensors or loading the checkpoint into memory.
    data_size = 10_000_000
    header = json.dumps(
        {"test": {"dtype": "F32", "shape": [data_size // 4], "data_offsets": [0, data_size]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        output.write(b"\0" * data_size)


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
        _write_valid_safetensors(tmp_path / candidates[0])
    assert validate_sd15_model(tmp_path) == tmp_path


def test_corrupt_component_weights_are_reported(tmp_path):
    for relative_path in REQUIRED_CONFIG_FILES:
        _write_nonempty(tmp_path / relative_path)
    for candidates in REQUIRED_WEIGHT_OPTIONS.values():
        _write_nonempty(tmp_path / candidates[0], b"not-a-safetensors-file" * 100)
    missing = missing_sd15_model_files(tmp_path)
    assert len([item for item in missing if "valid safetensors weights" in item]) == 3


def test_missing_model_directory_is_reported(tmp_path):
    missing_dir = tmp_path / "does-not-exist"
    assert missing_sd15_model_files(missing_dir) == ["<model directory>"]
    with pytest.raises(RuntimeError, match="model is incomplete"):
        validate_sd15_model(missing_dir)



def test_animatediff_adapter_requires_checkpoint_not_just_config(tmp_path):
    _write_nonempty(tmp_path / "config.json")
    missing = missing_animatediff_adapter_files(tmp_path)
    assert missing == ["<valid AnimateDiff safetensors weights>"]
    with pytest.raises(RuntimeError, match="AnimateDiff MotionAdapter is incomplete"):
        validate_animatediff_adapter(tmp_path)


def test_animatediff_adapter_passes_with_valid_weights(tmp_path):
    _write_nonempty(tmp_path / "config.json")
    _write_valid_safetensors(tmp_path / "diffusion_pytorch_model.safetensors")
    assert validate_animatediff_adapter(tmp_path) == tmp_path
