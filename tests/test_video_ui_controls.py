from pathlib import Path


def test_all_source_tabs_use_query_selector_all():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'const sourceTabs = $$(".source-tab");' in source
    assert 'sourceTabs.forEach(tab => tab.addEventListener("click", () => activateTrainingSourceTab(tab)));' in source
    assert 'const sourceTabs = $(".source-tab");' not in source


def test_generate_photo_and_strength_controls_have_handlers():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    template = Path("templates/generate_image.html").read_text(encoding="utf-8")
    assert 'generationForm?.addEventListener("submit",async event=>' in source
    assert "Reference strength is selected automatically" in template
    assert "strengthSlider" not in source
    assert "strengthNumber" not in source



def test_video_duration_and_quality_controls_are_wired_to_generation_form():
    template = Path("templates/generate_video.html").read_text(encoding="utf-8")
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'id="videoDurationSlider" type="range" min="1" max="60"' in template
    assert 'name="duration_seconds"' in template
    assert 'name="quality"' in template
    assert 'value="maximum"' in template
    assert 'updateVideoDurationControls' in source
    assert 'formData.set("task_id",taskId)' in source
    assert 'Base model (no trained face)' in template
