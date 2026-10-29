from pathlib import Path


def test_all_source_tabs_use_query_selector_all():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'const sourceTabs = $$(".source-tab");' in source
    assert 'sourceTabs.forEach(tab => tab.addEventListener("click", () => activateTrainingSourceTab(tab)));' in source
    assert 'const sourceTabs = $(".source-tab");' not in source


def test_generate_photo_and_strength_controls_have_handlers():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'strengthSlider?.addEventListener("input", () => syncStrength(strengthSlider, strengthNumber));' in source
    assert 'strengthNumber?.addEventListener("input", () => {' in source
    assert 'generationForm?.addEventListener("submit",async event=>' in source
