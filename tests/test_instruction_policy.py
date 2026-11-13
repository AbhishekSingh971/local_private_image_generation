import pytest

from instruction_policy import build_instruction_guidance
from prompt_compiler import build_structured_prompt


def test_reference_edit_preserves_unrequested_attributes():
    guidance = build_instruction_guidance(
        "Make the person smile",
        has_reference_image=True,
        has_trained_identity=True,
    )
    assert "selected trained person's identity" in guidance
    assert "original body pose" in guidance
    assert "original clothing" in guidance
    assert "original background" in guidance
    assert "original facial expression" not in guidance


def test_requested_pose_change_keeps_body_proportions():
    guidance = build_instruction_guidance(
        "Change the body pose to sitting",
        has_reference_image=True,
        has_trained_identity=True,
    )
    assert "original body pose" not in guidance
    assert "underlying body proportions" in guidance


def test_requested_body_shape_change_is_not_blocked():
    guidance = build_instruction_guidance(
        "Make the person's body shape more athletic",
        has_reference_image=True,
        has_trained_identity=True,
    )
    assert "Do not unintentionally change body shape" not in guidance


def test_requested_identity_change_is_not_blocked():
    guidance = build_instruction_guidance(
        "Replace the person with a different face",
        has_reference_image=True,
        has_trained_identity=True,
    )
    assert "selected trained person's identity" not in guidance


def test_clock_is_complete_by_default():
    guidance = build_instruction_guidance("A vintage clock on a desk")
    assert "properly formed dial" in guidance
    assert "correctly attached hour/minute hands" in guidance


def test_explicit_clock_hand_removal_is_respected():
    guidance = build_instruction_guidance("Remove the clock hands but keep the dial")
    assert "properly formed dial" in guidance
    assert "correctly attached hour/minute hands" not in guidance
    assert "Do not invent components the user explicitly asked to remove" in guidance


def test_removing_whole_clock_does_not_reintroduce_it():
    guidance = build_instruction_guidance("Remove the clock from the wall")
    assert "coherent clock or watch" not in guidance


def test_car_integrity_is_added_for_requested_car():
    guidance = build_instruction_guidance("A red car on a road")
    assert "aligned wheels" in guidance


def test_object_integrity_does_not_force_removed_component():
    guidance = build_instruction_guidance("A car with no wheels")
    assert "aligned wheels" not in guidance


def test_human_anatomy_is_supported_without_overriding_requested_removal():
    guidance = build_instruction_guidance("A person standing")
    assert "anatomically coherent human anatomy" in guidance


def test_prompt_compiler_applies_reference_and_identity_guidance():
    result = build_structured_prompt(
        "Make the person smile",
        {"expression": "gentle smile"},
        has_reference_image=True,
        has_trained_identity=True,
    )
    assert "Expression: gentle smile." in result
    assert "selected trained person's identity" in result


def test_prompt_compiler_keeps_original_prompt_and_explicit_controls():
    result = build_structured_prompt(
        "A clock on a table",
        {"lighting": "golden hour"},
    )
    assert result.startswith("A clock on a table")
    assert "Lighting: golden hour." in result
    assert "properly formed dial" in result
