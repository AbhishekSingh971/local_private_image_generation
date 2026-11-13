"""Pure compute-backend selection logic; hardware probing is performed by app.py."""

BACKENDS = ("cpu", "local_gpu", "cloud_gpu", "runpod")


def select_compute_backend(
    gpu_enabled: bool,
    requested: str | None,
    *,
    local_gpu_available: bool,
    cloud_gpu_available: bool,
) -> str:
    """Select CPU/local CUDA/cloud GPU after checking actual availability.

    ``runpod`` remains a legacy alias for ``cloud_gpu``. When acceleration is
    disabled, always use CPU. Automatic mode prefers a local CUDA GPU, then a
    reachable cloud worker, then CPU fallback.
    """
    if not gpu_enabled:
        return "cpu"

    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        if local_gpu_available:
            return "local_gpu"
        if cloud_gpu_available:
            return "cloud_gpu"
        return "cpu"

    if choice == "runpod":
        choice = "cloud_gpu"
    if choice not in BACKENDS:
        raise ValueError("Compute backend must be auto, cpu, local_gpu, or cloud_gpu.")
    if choice == "local_gpu" and not local_gpu_available:
        raise ValueError("System GPU is not available to PyTorch. Install compatible CUDA-enabled PyTorch and drivers.")
    if choice == "cloud_gpu" and not cloud_gpu_available:
        raise ValueError("Cloud GPU is not currently available. Check the worker URL/token and confirm the runtime is running.")
    return choice

def select_generation_backend(
    gpu_enabled: bool,
    requested: str | None,
    *,
    cloud_gpu_available: bool,
) -> str:
    """Route image/video generation to GHive first, otherwise local CPU.

    The local integrated GPU is intentionally not considered here. Local
    fallback is CPU-only and must use an already-present local SD 1.5 model.
    """
    if not gpu_enabled:
        return "cpu"

    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        return "cloud_gpu" if cloud_gpu_available else "cpu"
    if choice == "cpu":
        return "cpu"
    if choice in {"cloud_gpu", "runpod"}:
        if not cloud_gpu_available:
            raise ValueError(
                "GHive GPU is not currently available. Start the GHive worker "
                "or switch to local CPU for image generation."
            )
        return "cloud_gpu"
    if choice == "local_gpu":
        raise ValueError(
            "Image and video generation use the GHive GPU or local CPU. "
            "System GPU selection is not enabled for these workflows."
        )
    raise ValueError("Generation backend must be auto, cpu, or cloud_gpu.")

