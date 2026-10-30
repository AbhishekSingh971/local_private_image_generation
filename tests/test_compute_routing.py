import pytest

from compute_routing import select_compute_backend


def test_toggle_off_always_uses_cpu():
    assert select_compute_backend(
        False, "cloud_gpu", local_gpu_available=True, cloud_gpu_available=True
    ) == "cpu"
    assert select_compute_backend(
        False, "local_gpu", local_gpu_available=True, cloud_gpu_available=True
    ) == "cpu"


def test_auto_prefers_available_system_gpu():
    assert select_compute_backend(
        True, "auto", local_gpu_available=True, cloud_gpu_available=True
    ) == "local_gpu"


def test_auto_falls_back_to_cloud_gpu_when_system_gpu_is_unavailable():
    assert select_compute_backend(
        True, "auto", local_gpu_available=False, cloud_gpu_available=True
    ) == "cloud_gpu"


def test_cloud_gpu_selects_the_configured_ghive_worker():
    assert select_compute_backend(
        True, "cloud_gpu", local_gpu_available=False, cloud_gpu_available=True
    ) == "cloud_gpu"


def test_auto_falls_back_to_cpu_when_no_gpu_is_available():
    assert select_compute_backend(
        True, "auto", local_gpu_available=False, cloud_gpu_available=False
    ) == "cpu"


def test_explicit_unavailable_gpu_is_not_silently_selected():
    with pytest.raises(ValueError, match="System GPU is not available"):
        select_compute_backend(
            True, "local_gpu", local_gpu_available=False, cloud_gpu_available=True
        )
    with pytest.raises(ValueError, match="Cloud GPU is not currently available"):
        select_compute_backend(
            True, "cloud_gpu", local_gpu_available=True, cloud_gpu_available=False
        )
    with pytest.raises(ValueError, match="Compute backend must be"):
        select_compute_backend(
            True, "runpod", local_gpu_available=True, cloud_gpu_available=True
        )


def test_user_can_force_cpu_even_when_acceleration_is_enabled():
    assert select_compute_backend(
        True, "cpu", local_gpu_available=True, cloud_gpu_available=True
    ) == "cpu"


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="Compute backend must be"):
        select_compute_backend(
            True, "invented_gpu", local_gpu_available=True, cloud_gpu_available=True
        )
