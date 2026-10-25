"""Pure compute-backend selection logic; hardware probing is performed by app.py."""

BACKENDS = ("cpu", "local_gpu", "runpod")


def select_compute_backend(
    gpu_enabled: bool,
    requested: str | None,
    *,
    local_gpu_available: bool,
    cloud_gpu_available: bool,
) -> str:
    """Select CPU/local CUDA/RunPod after checking actual availability.

    With acceleration disabled, always use CPU. In automatic mode, prefer a
    local CUDA GPU, then a reachable cloud GPU, then CPU fallback. Explicit
    unavailable selections fail clearly instead of silently routing elsewhere.
    """
    if not gpu_enabled:
        return "cpu"

    choice = (requested or "auto").strip().lower()
    if choice == "auto":
        if local_gpu_available:
            return "local_gpu"
        if cloud_gpu_available:
            return "runpod"
        return "cpu"

    if choice not in BACKENDS:
        raise ValueError("Compute backend must be auto, cpu, local_gpu, or runpod.")
    if choice == "local_gpu" and not local_gpu_available:
        raise ValueError("System GPU is not available to PyTorch. Install compatible CUDA-enabled PyTorch and drivers.")
    if choice == "runpod" and not cloud_gpu_available:
        raise ValueError("RunPod GPU is not currently available. Check the worker URL/token and confirm the Pod is running.")
    return choice
