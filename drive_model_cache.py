"""Selective Google Drive -> local cache for Stable Diffusion 1.5.

Diffusers expects ordinary local files and random access to weight files. GHive
instances without FUSE cannot expose Google Drive as a POSIX filesystem, so this
module downloads only the inference files needed by SD 1.5 on worker startup.
The cache is reusable across worker restarts when its directory is persistent.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path

# The uploaded Drive directory contains a complete Diffusers snapshot nested
# inside "vae/". Its root is therefore .../stable-diffusion-v1-5/vae.
REQUIRED_FILES = (
    "model_index.json",
    "feature_extractor/preprocessor_config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

_LOCK = threading.Lock()


def _remote_root() -> str:
    return os.environ.get(
        "SD_MODEL_DRIVE_REMOTE",
        "gdrive:Local Ai/models/stable-diffusion-v1-5/vae",
    ).strip().rstrip("/")


def cache_dir() -> Path:
    configured = os.environ.get("SD_MODEL_CACHE_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / ".cache" / "local_private_image_generation" / "sd15").resolve()


def ensure_sd15_model_cached() -> Path:
    """Fetch missing inference files only; never copy the full Drive directory."""
    with _LOCK:
        root = cache_dir()
        root.mkdir(parents=True, exist_ok=True)
        remote = _remote_root()
        if not remote or ":" not in remote:
            raise RuntimeError(
                "SD_MODEL_DRIVE_REMOTE must be an rclone remote path, e.g. "
                "'gdrive:Local Ai/models/stable-diffusion-v1-5/vae'."
            )

        rclone = shutil.which("rclone")
        if not rclone:
            candidate = Path.home() / ".local" / "bin" / "rclone"
            if candidate.is_file():
                rclone = str(candidate)
            else:
                raise RuntimeError(
                    "rclone was not found. Install rclone and authenticate the 'gdrive' remote first."
                )

        missing = [name for name in REQUIRED_FILES if not (root / name).is_file() or (root / name).stat().st_size == 0]
        if not missing:
            print(f"Stable Diffusion 1.5 cache ready: {root}", flush=True)
            return root

        print(
            f"Google Drive model cache: {len(missing)} of {len(REQUIRED_FILES)} files missing. "
            "Only required Diffusers files will be fetched.",
            flush=True,
        )
        for relative in missing:
            source = f"{remote}/{relative}"
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".partial")
            temporary.unlink(missing_ok=True)
            print(f"Fetching {relative} ...", flush=True)
            command = [rclone, "copyto", source, str(temporary), "--progress", "--stats-one-line"]
            try:
                subprocess.run(command, check=True)
                if not temporary.is_file() or temporary.stat().st_size == 0:
                    raise RuntimeError(f"rclone produced an empty file for {relative}.")
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

        still_missing = [
            name for name in REQUIRED_FILES
            if not (root / name).is_file() or (root / name).stat().st_size == 0
        ]
        if still_missing:
            raise RuntimeError("Stable Diffusion cache is incomplete: " + ", ".join(still_missing))

        print(f"Stable Diffusion 1.5 cache complete: {root}", flush=True)
        return root


if __name__ == "__main__":
    print(ensure_sd15_model_cached())
