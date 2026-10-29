from pathlib import Path


def test_source_tabs_use_all_matching_elements():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'const $$ = (selector, root=document) => [...root.querySelectorAll(selector)];' in source
    assert 'const sourceTabs = $$(".source-tab");' in source
    assert 'sourceTabs.forEach(tab => tab.addEventListener("click", () => activateTrainingSourceTab(tab)));' in source
    assert 'const sourceTabs = $(".source-tab");' not in source
