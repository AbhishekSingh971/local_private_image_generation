"""Download only the required Diffusers SD 1.5 model assets and verify them."""
import argparse
import os
from pathlib import Path

from model_integrity import SD15_ALLOW_PATTERNS, missing_sd15_model_files, validate_sd15_model

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "models" / "stable-diffusion-v1-5"
DEFAULT_REPO_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def ensure_model(model_dir, repo_id=None):
    model_dir = Path(model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = (ROOT / model_dir).resolve()

    missing = missing_sd15_model_files(model_dir)
    if not missing:
        print(f"Stable Diffusion 1.5 model verified: {model_dir}")
        return model_dir

    repo_id = (repo_id or os.environ.get("SD_MODEL_ID") or DEFAULT_REPO_ID).strip()
    if not repo_id:
        raise RuntimeError("SD_MODEL_ID is empty; cannot download the Stable Diffusion base model.")

    print(f"Incomplete SD 1.5 model detected at: {model_dir}")
    print("Missing/invalid assets:")
    for item in missing:
        print(f"  - {item}")
    print("Resuming the required Diffusers checkpoint download. Existing files are kept.")

    from huggingface_hub import snapshot_download

    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=list(SD15_ALLOW_PATTERNS),
    )

    # Validate again after download; never start the worker with metadata-only weights.
    validate_sd15_model(model_dir)
    print(f"Stable Diffusion 1.5 model is complete: {model_dir}")
    return model_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=os.environ.get("SD_MODEL_PATH") or str(DEFAULT_MODEL_DIR),
        help="Diffusers model directory (default: models/stable-diffusion-v1-5).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the model without downloading missing files.",
    )
    args = parser.parse_args()

    if args.check_only:
        validate_sd15_model(args.model_dir)
        print(f"Stable Diffusion 1.5 model verified: {Path(args.model_dir).expanduser().resolve()}")
        return

    ensure_model(args.model_dir)


if __name__ == "__main__":
    main()
