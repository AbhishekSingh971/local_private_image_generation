from pathlib import Path


def test_all_source_tabs_use_query_selector_all():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'const sourceTabs = $$(".source-tab");' in source
    assert 'sourceTabs.forEach(tab => tab.addEventListener("click", () => activateTrainingSourceTab(tab)));' in source
    assert 'const sourceTabs = $(".source-tab");' not in source


def test_generate_photo_uses_prompt_first_form_without_manual_strength_controls():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    template = Path("templates/generate_image.html").read_text(encoding="utf-8")
    assert 'generationForm?.addEventListener("submit",async event=>' in source
    assert 'name="strength" id="strength" value="0.45"' in template
    assert 'id="strengthNumber"' not in template
    assert 'id="strengthGroup"' not in template
    assert 'name="prompt"' in template
    assert 'name="style"' in template
    assert 'name="composition"' in template
    assert 'name="camera"' in template
    assert 'name="lighting"' in template
    assert 'id="generationProfile"' in template
    assert 'id="imagePreset"' in template
    assert 'id="faceAngle" name="face_angle"' in template
    assert 'id="faceExpression" name="face_expression"' in template
    assert 'id="steps" name="steps" type="number" min="5" max="35"' in template
    assert 'sd15: { min: 5, max: 35, default: 25 }' in source
    assert 'sdxl: { min: 5, max: 30, default: 25 }' in source
    assert 'flux2_klein: { min: 4, max: 4, default: 4 }' in source
    assert 'const sd15Selected = (modelFamilySelect?.value || "sd15") === "sd15";' in source
    assert 'const controlsEnabled = profileSelected && sd15Selected;' in source



def test_local_and_rented_gpu_generation_share_step_validation():
    app_source = Path("app.py").read_text(encoding="utf-8")
    worker_source = Path("gpu_worker.py").read_text(encoding="utf-8")
    assert "steps = resolve_image_inference_steps(request.form.get(\"steps\"), model_family)" in app_source
    assert "steps = local_engine.resolve_image_inference_steps(request.form.get(\"steps\"), model_family)" in worker_source



def test_training_photo_edit_preview_uses_sd15_limit():
    app_source = Path("app.py").read_text(encoding="utf-8")
    template = Path("templates/training_data.html").read_text(encoding="utf-8")
    assert 'steps = resolve_image_inference_steps(request.form.get("steps"), "sd15")' in app_source
    assert '<option value="30">30</option>' in template
    assert '<option value="35">35 · maximum for SD 1.5</option>' in template


def test_face_angle_live_status_shows_backend_and_failed_target_error():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'active.backend==="cloud_gpu"||active.backend==="runpod"?"Cloud GPU"' in source
    assert 'String(latestFailure.error||"Unknown error").slice(0,500)' in source
    assert '· "+failedCount+" failed."' in source


def test_face_training_ui_offers_lora_and_full_finetuning():
    template = Path("templates/faces.html").read_text(encoding="utf-8")
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'name="training_mode"' in template
    assert 'value="lora"' in template
    assert 'value="full_finetune"' in template
    assert 'Full U-Net fine-tuning: 100–2000 steps.' in source
    assert 'Full fine-tuning updates every SD 1.5 U-Net weight' in source



def test_shared_body_adapter_is_opt_in_for_face_identity_generation():
    template = Path("templates/generate_image.html").read_text(encoding="utf-8")
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert '<input type="checkbox" name="use_global_body_model" id="useGlobalBodyModel" value="true">' in template
    assert 'type="hidden" name="use_global_body_model"' not in template
    assert "optional · off by default" in template
    assert "Shared body-type conditioning is optional and stays off by default" in source


def test_generation_metadata_reports_actual_identity_adapter_state():
    app_source = Path("app.py").read_text(encoding="utf-8")
    worker_source = Path("gpu_worker.py").read_text(encoding="utf-8")
    assert '"face_adapter_loaded": bool(getattr(pipeline, "_private_face_lora_loaded", False))' in app_source
    assert '"body_adapter_loaded": bool(getattr(pipeline, "_private_body_lora_loaded", False))' in app_source
    assert 'response.headers["X-Identity-Conditioning"]' in worker_source
    assert 'identity_conditioning=identity_conditioning' in app_source



def test_generation_status_displays_face_adapter_activation_report():
    source = Path("static/studio.js").read_text(encoding="utf-8")
    assert 'identity.training_mode==="full_finetune"?"full fine-tune"' in source
    assert 'identity.face_adapter_loaded?"face LoRA loaded":"FACE LoRA NOT LOADED"' in source
    assert 'identity.trigger_token_applied?"applied":"missing"' in source
    assert 'identity.body_adapter_loaded?"on":"off"' in source
