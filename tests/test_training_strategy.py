import pytest

from training.strategy import build_training_plan


def test_lora_plan_uses_existing_trainer_when_requirements_are_met():
    plan = build_training_plan(
        "lora", base_model="sd15", profile_id="a" * 32,
        dataset_path="data/face_profiles/example/dataset",
        cuda_available=True, dataset_image_count=8,
    )
    assert plan.executable is True
    assert plan.trainer_entrypoint == "train_face_lora.py"


def test_full_finetune_never_falls_back_to_lora():
    plan = build_training_plan(
        "full_finetune", base_model="sd15", profile_id=None,
        dataset_path="data/domain/dataset",
        cuda_available=True, dataset_image_count=100,
    )
    assert plan.executable is False
    assert plan.trainer_entrypoint is None
    assert any("not implemented" in item for item in plan.blockers)


def test_cpu_training_is_blocked():
    plan = build_training_plan(
        "lora", base_model="sd15", profile_id="a" * 32,
        dataset_path="dataset", cuda_available=False, dataset_image_count=8,
    )
    assert not plan.executable
    assert any("CUDA" in item for item in plan.blockers)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        build_training_plan(
            "dreambooth", base_model="sd15", profile_id=None,
            dataset_path="dataset", cuda_available=True, dataset_image_count=10,
        )
