"""Download and verify the AnimateDiff SD 1.5 motion adapter."""
import argparse
import os
from pathlib import Path

from model_integrity import (
    ANIMATEDIFF_ALLOW_PATTERNS,
    missing_animatediff_adapter_files,
    validate_animatediff_adapter,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_ADAPTER_DIR = ROOT / "models" / "animatediff-motion-adapter-v1-5-2"
DEFAULT_REPO_ID = "guoyww/animatediff-motion-adapter-v1-5-2"


def ensure_adapter(adapter_dir, repo_id=None):
    adapter_dir = Path(adapter_dir).expanduser()
    if not adapter_dir.is_absolute():
        adapter_dir = (ROOT / adapter_dir).resolve()

    missing = missing_animatediff_adapter_files(adapter_dir)
    if not missing:
        print(f"AnimateDiff motion adapter verified: {adapter_dir}")
        return adapter_dir

    repo_id = (repo_id or os.environ.get("ANIMATEDIFF_MODEL_ID") or DEFAULT_REPO_ID).strip()
    if not repo_id:
        raise RuntimeError("ANIMATEDIFF_MODEL_ID is empty; cannot download the motion adapter.")

    print(f"Incomplete AnimateDiff motion adapter detected at: {adapter_dir}")
    print("Missing/invalid assets:")
    for item in missing:
        print(f"  - {item}")
    print("Resuming the adapter checkpoint download. Existing files are kept.")

    # Avoid Xet streaming failures on unstable GHive network paths and
    # serialize large-file downloads to make resume behavior more reliable.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    adapter_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(adapter_dir),
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=list(ANIMATEDIFF_ALLOW_PATTERNS),
        max_workers=1,
    )

    validate_animatediff_adapter(adapter_dir)
    print(f"AnimateDiff motion adapter is complete: {adapter_dir}")
    return adapter_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter-dir",
        default=str(DEFAULT_ADAPTER_DIR),
        help="AnimateDiff MotionAdapter directory.",
    )
    parser.add_argument("--check-only", action="store_true", help="Validate without downloading.")
    args = parser.parse_args()

    if args.check_only:
        validate_animatediff_adapter(args.adapter_dir)
        print(f"AnimateDiff motion adapter verified: {Path(args.adapter_dir).expanduser().resolve()}")
        return

    ensure_adapter(args.adapter_dir)


if __name__ == "__main__":
    main()
