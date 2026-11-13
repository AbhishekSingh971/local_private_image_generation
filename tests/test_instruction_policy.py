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


def test_descriptive_reference_details_are_not_mistaken_for_edit_requests():
    guidance = build_instruction_guidance(
        "A woman wearing a red dress, standing in a room",
        has_reference_image=True,
    )
    assert "original clothing" in guidance
    assert "original body pose" in guidance
    assert "original background" in guidance


def test_change_background_does_not_release_outfit_preservation():
    guidance = build_instruction_guidance(
        "Change the background to a beach but keep the red dress",
        has_reference_image=True,
    )
    assert "original background" not in guidance
    assert "original clothing" in guidance


def test_remove_hands_before_clock_does_not_remove_whole_clock():
    guidance = build_instruction_guidance(
        "Remove the hands from the clock",
    )
    assert "properly formed dial" in guidance
    assert "correctly attached hour/minute hands" not in guidance


def test_negated_change_is_not_treated_as_a_requested_change():
    guidance = build_instruction_guidance(
        "Do not change the clothing",
        has_reference_image=True,
    )
    assert "original clothing" in guidance


def test_expression_change_does_not_release_trained_identity():
    guidance = build_instruction_guidance(
        "Change the person's expression to a smile",
        has_reference_image=True,
        has_trained_identity=True,
    )
    assert "selected trained person's identity" in guidance
    assert "original facial expression" not in guidance


def test_trained_identity_is_preserved_without_reference_image():
    guidance = build_instruction_guidance(
        "A portrait of the trained person smiling",
        has_trained_identity=True,
    )
    assert "selected trained person's identity" in guidance


def test_digital_clock_gets_display_not_analog_hands():
    guidance = build_instruction_guidance("A digital clock on a bedside table")
    assert "digital time display" in guidance
    assert "hour/minute hands" not in guidance
