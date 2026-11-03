"""Validate the local Stable Diffusion 1.5 Diffusers model assets.

The app and trainer use the same on-disk model directory. A model_index.json by
itself is not proof that the component weights were downloaded successfully.
"""
from pathlib import Path

REQUIRED_CONFIG_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "text_encoder/config.json",
    "unet/config.json",
    "vae/config.json",
)

# Diffusers component checkpoints are stored as safetensors in the official
# SD 1.5 repository. The fp16 variants are accepted for existing installations.
REQUIRED_WEIGHT_OPTIONS = {
    "text_encoder": (
        "text_encoder/model.safetensors",
        "text_encoder/model.fp16.safetensors",
    ),
    "unet": (
        "unet/diffusion_pytorch_model.safetensors",
        "unet/diffusion_pytorch_model.fp16.safetensors",
    ),
    "vae": (
        "vae/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.fp16.safetensors",
    ),
}
MIN_WEIGHT_BYTES = 10_000_000


def missing_sd15_model_files(model_dir):
    """Return required model paths that are absent, empty or invalid."""
    root = Path(model_dir).expanduser()
    missing = []

    if not root.is_dir():
        return ["<model directory>"]

    for relative_path in REQUIRED_CONFIG_FILES:
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(relative_path)

    try:
        from safetensors import safe_open
    except ImportError:
        safe_open = None

    for component, candidates in REQUIRED_WEIGHT_OPTIONS.items():
        valid_weight = None
        for relative_path in candidates:
            path = root / relative_path
            if not path.is_file():
                continue
            try:
                if path.stat().st_size < MIN_WEIGHT_BYTES:
                    continue
                if safe_open is not None:
                    with safe_open(str(path), framework="pt", device="cpu") as handle:
                        if not next(iter(handle.keys()), None):
                            continue
                valid_weight = path
                break
            except Exception:
                continue
        if valid_weight is None:
            missing.append(f"{component}/<valid safetensors weights>")

    return missing


def validate_sd15_model(model_dir):
    """Raise a detailed error when a Diffusers SD 1.5 model is incomplete."""
    root = Path(model_dir).expanduser()
    missing = missing_sd15_model_files(root)
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise RuntimeError(
            "Stable Diffusion 1.5 model is incomplete or contains invalid weights at "
            f"{root.resolve()}. Missing/invalid assets:\n{details}\n"
            "Run 'python ensure_sd15_model.py' on the GPU worker to resume the "
            "required model download. Do not delete the existing model directory."
        )
    return root


ANIMATEDIFF_WEIGHT_OPTIONS = (
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
)


def missing_animatediff_adapter_files(adapter_dir):
    """Return missing or invalid AnimateDiff MotionAdapter assets."""
    root = Path(adapter_dir).expanduser()
    if not root.is_dir():
        return ["<AnimateDiff adapter directory>"]

    missing = []
    config_path = root / "config.json"
    if not config_path.is_file() or config_path.stat().st_size == 0:
        missing.append("config.json")

    try:
        from safetensors import safe_open
    except ImportError:
        safe_open = None

    valid_weight = False
    for filename in ANIMATEDIFF_WEIGHT_OPTIONS:
        path = root / filename
        if not path.is_file():
            continue
        try:
            if path.stat().st_size < MIN_WEIGHT_BYTES:
                continue
            if safe_open is not None:
                with safe_open(str(path), framework="pt", device="cpu") as handle:
                    if not next(iter(handle.keys()), None):
                        continue
            valid_weight = True
            break
        except Exception:
            continue

    if not valid_weight:
        missing.append("<valid AnimateDiff safetensors weights>")
    return missing


def validate_animatediff_adapter(adapter_dir):
    """Raise when the AnimateDiff adapter config or checkpoint is incomplete."""
    root = Path(adapter_dir).expanduser()
    missing = missing_animatediff_adapter_files(root)
    if missing:
        details = "\\n".join(f"  - {item}" for item in missing)
        raise RuntimeError(
            f"AnimateDiff MotionAdapter is incomplete or invalid at {root.resolve()}. "
            f"Missing/invalid assets:\\n{details}\\n"
            "Run 'python ensure_animatediff_adapter.py' on the GPU worker to resume "
            "the adapter download. Do not delete the existing adapter directory."
        )
    return root


ANIMATEDIFF_ALLOW_PATTERNS = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.safetensors",
)


SD15_ALLOW_PATTERNS = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
)
