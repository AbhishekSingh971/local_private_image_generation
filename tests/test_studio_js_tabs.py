from pathlib import Path


def test_source_tabs_use_all_matching_elements():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];' in source
    assert 'const sourceTabs = $$(".source-tab");' in source
    assert 'sourceTabs.forEach(tab => tab.addEventListener("click", () => activateTrainingSourceTab(tab)));' in source
    assert 'const sourceTabs = $(".source-tab");' not in source


def test_angle_and_expression_targets_keep_separate_selection_state():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert "const selectedFaceTargetsByMode={" in source
    assert "face_angles:new Set(faceVariationLabels.face_angles)" in source
    assert "expressions:new Set(faceVariationLabels.expressions)" in source
    assert "checkbox.checked=selected.has(label);" in source
    assert "selectedFaceTargetsByMode[activeVariationMode]=new Set(checked.map(box=>box.value));" in source
