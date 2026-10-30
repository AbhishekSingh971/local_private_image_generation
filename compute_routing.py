"""Pure compute-backend selection logic; hardware probing is performed by app.py."""

BACKENDS = ("cpu", "local_gpu", "cloud_gpu")


def select_compute_backend(
    gpu_enabled: bool,
    requested: str | None,
    *,
    local_gpu_available: bool,
    cloud_gpu_available: bool,
) -> str:
    """Select CPU/local CUDA/cloud GPU after checking actual availability.

    The cloud GPU backend is the configured GHive worker. When acceleration is
    disabled, always use CPU. Automatic mode prefers a local CUDA GPU, then a
    reachable GHive worker, then CPU fallback.
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

    if choice not in BACKENDS:
        raise ValueError("Compute backend must be auto, cpu, local_gpu, or cloud_gpu.")
    if choice == "local_gpu" and not local_gpu_available:
        raise ValueError("System GPU is not available to PyTorch. Install compatible CUDA-enabled PyTorch and drivers.")
    if choice == "cloud_gpu" and not cloud_gpu_available:
        raise ValueError("Cloud GPU is not currently available. Check the worker URL/token and confirm the runtime is running.")
    return choice
