"""Authenticated GPU worker for a user-managed Colab or cloud GPU runtime.

Set REMOTE_API_TOKEN to a strong random secret and expose port 8000 only through
an HTTPS tunnel/proxy. This worker does not provision, purchase, or keep a runtime alive.
"""
import hmac
import io
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import zipfile
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

import app as local_engine
from image_generation import build_generation_prompts, resolve_generation_seed, resolve_image_dimensions, validate_image_preset
from video_generation import validate_video_settings
from face_profiles import (
    create_profile,
    dataset_images,
    get_profile,
    model_path,
    save_profile_image,
    update_profile,
    validate_profile_id,
)

worker = Flask("runpod_gpu_worker")
worker.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
TOKEN = os.environ.get("REMOTE_API_TOKEN", "").strip()
TASKS = {}
TASKS_LOCK = threading.RLock()
GENERATION_PROGRESS = {}
GENERATION_PROGRESS_LOCK = threading.RLock()
GENERATION_CANCEL_REQUESTS = set()
GENERATION_CANCEL_LOCK = threading.RLock()
REMOTE_GENERATION_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="remote-image-generation")
REMOTE_GENERATION_JOBS = {}
REMOTE_GENERATION_JOBS_LOCK = threading.RLock()


class RemoteGenerationCancelled(RuntimeError):
    pass


def _set_generation_progress(task_id, *, asset_type, status, progress, current_step=0, total_steps=0, stage="", message=""):
    if not re.fullmatch(r"[a-f0-9]{32}", str(task_id or "")):
        return
    now = time.time()
    record = {
        "task_id": task_id, "asset_type": asset_type, "backend": "runpod",
        "status": status, "progress": max(0, min(100, int(progress))),
        "current_step": max(0, int(current_step or 0)),
        "total_steps": max(0, int(total_steps or 0)),
        "stage": str(stage or "")[:80], "message": str(message or "")[:300],
        "updated_at": now,
    }
    with GENERATION_PROGRESS_LOCK:
        previous = GENERATION_PROGRESS.get(task_id, {})
        record["created_at"] = previous.get("created_at", now)
        GENERATION_PROGRESS[task_id] = record
        while len(GENERATION_PROGRESS) > 200:
            oldest = next(iter(GENERATION_PROGRESS))
            if oldest == task_id and len(GENERATION_PROGRESS) > 1:
                oldest = next(key for key in GENERATION_PROGRESS if key != task_id)
            GENERATION_PROGRESS.pop(oldest, None)

TRAIN_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="runpod-face-train")
VIDEO_PIPE = None
VIDEO_ACTIVE_LORA = None
VIDEO_LOCK = threading.Lock()


@worker.before_request
def require_worker_token():
    expected = os.environ.get("REMOTE_API_TOKEN", "").strip()
    supplied = request.headers.get("Authorization", "")
    if not expected:
        return jsonify(error="REMOTE_API_TOKEN is not configured on the GPU worker."), 503
    if not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:].strip(), expected):
        return jsonify(error="Unauthorized."), 401
    return None


@worker.get("/api/remote/generation-progress/<task_id>")
def remote_generation_progress(task_id):
    if not re.fullmatch(r"[a-f0-9]{32}", task_id or ""):
        return jsonify(error="Invalid generation task ID."), 400
    with GENERATION_PROGRESS_LOCK:
        progress = GENERATION_PROGRESS.get(task_id)
        if progress:
            return jsonify(dict(progress))
    return jsonify(
        task_id=task_id, asset_type="unknown", backend="runpod", status="starting",
        progress=0, current_step=0, total_steps=0, stage="starting",
        message="Waiting for the GPU worker…",
    )


@worker.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Request too large. Keep each request under 64 MB."), 413


def _valid_profile(profile_id):
    try:
        value = validate_profile_id(profile_id)
    except ValueError:
        return None
    return value if get_profile(value) else None


def _training_job(profile_id, job_id, steps, resolution, purpose="identity"):
    profile = get_profile(profile_id)
    if not profile:
        with TASKS_LOCK:
            TASKS[job_id] = {"job_id": job_id, "profile_id": profile_id, "status": "failed", "error": "Profile not found."}
        return

    command = [
        sys.executable, str(ROOT / "train_face_lora.py"),
        "--profile-id", profile_id,
        "--output-name", profile_id,
        "--token", profile["token"],
        "--steps", str(steps),
        "--resolution", str(resolution),
        "--mixed-precision", "fp16",
        "--dataset-purpose", purpose,
    ]
    with TASKS_LOCK:
        TASKS[job_id] = {
            "job_id": job_id, "profile_id": profile_id, "purpose": purpose, "status": "running",
            "started_at": time.time(), "logs": "",
        }
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=8 * 60 * 60,
            check=False,
        )
        logs = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        output_dir = (model_path(profile_id) if purpose == "identity" or profile_id == local_engine.GLOBAL_BODY_PROFILE_ID
                      else model_path(profile_id) / "components" / purpose)
        weights_exist = ((output_dir / "pytorch_lora_weights.safetensors").is_file()
                         or (output_dir / "pytorch_lora_weights.bin").is_file())
        if completed.returncode == 0 and weights_exist:
            if purpose == "identity" or profile_id == local_engine.GLOBAL_BODY_PROFILE_ID:
                update_profile(
                    profile_id, local_model_ready=True, local_training_state="completed",
                    local_training_error=None,
                )
            task = {"job_id": job_id, "profile_id": profile_id, "purpose": purpose, "status": "completed", "logs": logs[-5000:]}
        else:
            error = logs[-3500:] or f"Training exited with code {completed.returncode}."
            if purpose == "identity" or profile_id == local_engine.GLOBAL_BODY_PROFILE_ID:
                update_profile(
                    profile_id, local_training_state="failed", local_training_error=error,
                    local_model_ready=False,
                )
            task = {"job_id": job_id, "profile_id": profile_id, "purpose": purpose, "status": "failed", "error": error, "logs": logs[-5000:]}
    except Exception as exc:
        error = str(exc)
        if purpose == "identity" or profile_id == local_engine.GLOBAL_BODY_PROFILE_ID:
            update_profile(
                profile_id, local_training_state="failed", local_training_error=error,
                local_model_ready=False,
            )
        task = {"job_id": job_id, "profile_id": profile_id, "purpose": purpose, "status": "failed", "error": error}
    with TASKS_LOCK:
        task["finished_at"] = time.time()
        TASKS[job_id] = task


@worker.get("/api/remote/health")
def remote_health():
    if not torch.cuda.is_available():
        return jsonify(status="error", cuda_available=False, error="CUDA is not available in this Pod."), 503
    gpu = torch.cuda.get_device_properties(0)
    return jsonify(
        status="ok",
        cuda_available=True,
        gpu_name=torch.cuda.get_device_name(0),
        vram_gb=round(gpu.total_memory / (1024 ** 3), 2),
        face_profiles_ready=sum(
            1 for p in local_engine.list_profiles() if p.get("local_model_ready")
        ),
    )


@worker.post("/api/remote/profiles/<profile_id>/lora")
def remote_upload_profile_lora(profile_id):
    """Install a validated local safetensors LoRA under its existing profile ID."""
    try:
        profile_id = validate_profile_id(profile_id)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    uploaded = request.files.get("lora")
    if not uploaded or Path(uploaded.filename or "").name != "pytorch_lora_weights.safetensors":
        return jsonify(error="Upload pytorch_lora_weights.safetensors as the lora file."), 400
    raw = uploaded.stream.read(60 * 1024 * 1024 + 1)
    if not raw or len(raw) > 60 * 1024 * 1024:
        return jsonify(error="LoRA file must be non-empty and no larger than 60 MB."), 413

    # Parse the safetensors header without unpickling arbitrary Python objects.
    try:
        from safetensors import safe_open
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as temp_file:
            temp_file.write(raw)
            temp_file.flush()
            with safe_open(temp_file.name, framework="pt", device="cpu") as model_file:
                if not model_file.keys():
                    raise ValueError("No tensors found.")
    except Exception:
        return jsonify(error="Uploaded LoRA is not a valid safetensors model."), 400

    config_upload = request.files.get("adapter_config")
    config_bytes = None
    if config_upload and config_upload.filename:
        if Path(config_upload.filename).name != "adapter_config.json":
            return jsonify(error="Only adapter_config.json is accepted as optional metadata."), 400
        config_bytes = config_upload.stream.read(256 * 1024 + 1)
        if len(config_bytes) > 256 * 1024:
            return jsonify(error="Adapter config exceeds the 256 KB limit."), 413
        try:
            config = json.loads(config_bytes.decode("utf-8"))
            if not isinstance(config, dict):
                raise ValueError("Expected a JSON object.")
        except (UnicodeError, ValueError):
            return jsonify(error="adapter_config.json is invalid JSON."), 400

    purpose = (request.form.get("purpose") or "identity").strip().lower()
    if purpose not in {"identity", "body", "angle", "expression"}:
        return jsonify(error="Adapter purpose must be identity, body, angle or expression."), 400
    profile = get_profile(profile_id)
    if not profile:
        try:
            create_profile(
                (request.form.get("name") or "Synced face profile").strip()[:80],
                profile_id=profile_id,
                token=(request.form.get("token") or "").strip() or None,
                description=request.form.get("description") or "",
            )
        except (ValueError, RuntimeError) as exc:
            return jsonify(error=str(exc)), 400

    destination = (
        model_path(profile_id)
        if purpose == "identity" or (purpose == "body" and profile_id == local_engine.GLOBAL_BODY_PROFILE_ID)
        else model_path(profile_id) / "components" / purpose
    )
    destination.mkdir(parents=True, exist_ok=True)
    weight_path = destination / "pytorch_lora_weights.safetensors"
    config_path = destination / "adapter_config.json"
    temp_weight = destination / ".pytorch_lora_weights.safetensors.uploading"
    temp_config = destination / ".adapter_config.json.uploading"
    try:
        temp_weight.write_bytes(raw)
        if config_bytes is not None:
            temp_config.write_bytes(config_bytes)
        os.replace(temp_weight, weight_path)
        if config_bytes is not None:
            os.replace(temp_config, config_path)
        if purpose == "identity" or profile_id == local_engine.GLOBAL_BODY_PROFILE_ID:
            update_profile(
                profile_id, local_model_ready=True, local_training_state="completed",
                local_training_error=None,
            )
    except Exception as exc:
        temp_weight.unlink(missing_ok=True)
        temp_config.unlink(missing_ok=True)
        weight_path.unlink(missing_ok=True)
        return jsonify(error=f"Could not install LoRA on worker: {str(exc)[:500]}"), 500

    return jsonify(status="installed", ready=True, profile_id=profile_id, purpose=purpose, bytes=len(raw))


@worker.post("/api/remote/train-profile")
def remote_train_profile():
    if not torch.cuda.is_available():
        return jsonify(error="This cloud GPU worker has no CUDA GPU attached."), 503
    try:
        profile_id = validate_profile_id(request.form.get("profile_id", ""))
        name = (request.form.get("name") or "").strip()
        token = (request.form.get("token") or "").strip()
        description = request.form.get("description") or ""
        steps = int(request.form.get("steps", "800"))
        resolution = int(request.form.get("resolution", "512"))
        purpose = (request.form.get("purpose") or ("body" if profile_id == local_engine.GLOBAL_BODY_PROFILE_ID else "identity")).strip().lower()
        if purpose not in {"identity", "body", "expression", "angle"}:
            raise ValueError("Training purpose must be identity, body, expression or angle.")
        if profile_id == local_engine.GLOBAL_BODY_PROFILE_ID and purpose != "body":
            raise ValueError("The reserved shared profile can only train the body component.")
        if not 100 <= steps <= 3000 or resolution not in (384, 512):
            raise ValueError("Training steps must be 100–3000 and resolution must be 384 or 512.")
        photo_uploads = request.files.getlist("photos")
        images = local_engine._parse_uploads(photo_uploads)
        if len(images) < 5:
            raise ValueError("At least 5 valid face photos are required.")
        caption_by_name = {}
        for caption_file in request.files.getlist("captions"):
            try:
                caption_by_name[Path(caption_file.filename or "").name] = caption_file.stream.read(1200).decode("utf-8", errors="replace").strip()
            except (OSError, UnicodeError):
                continue
        existing_profile = get_profile(profile_id)
        if existing_profile:
            if existing_profile.get("token") != token:
                raise ValueError("The existing GPU-worker profile token does not match this training request.")
            # Repeated cloud training requests commonly resend the same complete
            # dataset. Do not duplicate an already-restored dataset on the worker.
            existing_eligible = local_engine._eligible_profile_training_images(profile_id, purpose)
            should_import = len(existing_eligible) < 5
        else:
            create_profile(name, profile_id=profile_id, token=token, description=description)
            should_import = True
        if should_import:
            for upload, image in zip(photo_uploads, images):
                caption = caption_by_name.get(Path(upload.filename or "").name, "")
                if caption:
                    lowered = caption.lower()
                    if lowered.startswith("body type:") or lowered.startswith("body proportions:"):
                        photo_kind = "body"
                    elif any(term in lowered for term in ("smile", "laugh", "expression", "surprised", "neutral", "serious", "sad", "angry", "fearful", "disgusted", "confused")):
                        photo_kind = "expression"
                    elif any(term in lowered for term in ("yaw", "pitch", "profile", "three-quarter", "front-facing", "head pose", "head turned")):
                        photo_kind = "angle"
                    else:
                        photo_kind = "manual"
                    save_profile_image(profile_id, image, angle_label=caption, photo_kind=photo_kind)
                else:
                    save_profile_image(profile_id, image, photo_kind="manual")
        eligible = local_engine._eligible_profile_training_images(profile_id, purpose)
        if len(eligible) < 5:
            raise ValueError(f"Found {len(eligible)} eligible {purpose} photos on the worker; at least 5 are required.")
        existing = get_profile(profile_id)
        if existing and existing.get("local_training_state") == "running" and (purpose == "identity" or profile_id == local_engine.GLOBAL_BODY_PROFILE_ID):
            return jsonify(error="This face profile is already training on the GPU worker."), 409
        job_id = uuid.uuid4().hex
        if purpose == "identity" or profile_id == local_engine.GLOBAL_BODY_PROFILE_ID:
            update_profile(
                profile_id, local_training_state="running", local_training_job_id=job_id,
                local_training_error=None, local_model_ready=False,
            )
        with TASKS_LOCK:
            TASKS[job_id] = {"job_id": job_id, "profile_id": profile_id, "purpose": purpose, "status": "queued"}
        TRAIN_POOL.submit(_training_job, profile_id, job_id, steps, resolution, purpose)
        return jsonify(status="queued", job_id=job_id, profile_id=profile_id, purpose=purpose), 202
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 500


@worker.get("/api/remote/jobs/<job_id>")
def remote_job_status(job_id):
    with TASKS_LOCK:
        task = TASKS.get(job_id)
    if not task:
        return jsonify(error="Training job not found. The GPU worker may have restarted; check the profile model files."), 404
    return jsonify(task)


@worker.get("/api/remote/models/<profile_id>")
def remote_download_model(profile_id):
    """Return the selected profile's LoRA weights as an authenticated ZIP bundle."""
    profile_id = _valid_profile(profile_id)
    if not profile_id:
        return jsonify(error="Face profile not found."), 404
    folder = model_path(profile_id)
    if not folder.is_dir():
        return jsonify(error="Trained model files were not found on this worker."), 404
    files = [item for item in folder.rglob("*") if item.is_file()]
    if not files:
        return jsonify(error="Trained model files were not found on this worker."), 404
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            archive.write(item, arcname=item.relative_to(folder).as_posix())
    output.seek(0)
    response = send_file(
        output, mimetype="application/zip", as_attachment=True,
        download_name=f"{profile_id}-lora.zip",
    )
    response.headers["X-Model-File-Count"] = str(len(files))
    return response


@worker.post("/api/remote/generation-cancel/<task_id>")
def remote_generation_cancel(task_id):
    if not re.fullmatch(r"[a-f0-9]{32}", task_id or ""):
        return jsonify(error="Invalid generation task ID."), 400
    with GENERATION_CANCEL_LOCK:
        GENERATION_CANCEL_REQUESTS.add(task_id)
    with GENERATION_PROGRESS_LOCK:
        current = dict(GENERATION_PROGRESS.get(task_id) or {})
    if current and current.get("status") not in {"completed", "failed", "cancelled"}:
        _set_generation_progress(
            task_id, asset_type=current.get("asset_type", "image"), status="cancelling",
            progress=current.get("progress", 0), current_step=current.get("current_step", 0),
            total_steps=current.get("total_steps", 0), stage="cancelling",
            message="Cancellation requested; stopping at the next denoising step.",
        )
    return jsonify(status="cancelling", task_id=task_id, message="Cancellation request recorded."), 202


def _remote_generation_cancelled(task_id):
    with GENERATION_CANCEL_LOCK:
        return task_id in GENERATION_CANCEL_REQUESTS


def _run_async_remote_generation(job_id, form_data, file_data):
    """Run the existing synchronous image route in a worker thread and retain its result."""
    with REMOTE_GENERATION_JOBS_LOCK:
        job = REMOTE_GENERATION_JOBS.get(job_id)
        if not job:
            return
        job["status"] = "queued"
    try:
        # Serialize queued image jobs with video and legacy synchronous image requests.
        while local_engine.generation_lock.locked():
            if _remote_generation_cancelled(job_id):
                with REMOTE_GENERATION_JOBS_LOCK:
                    job = REMOTE_GENERATION_JOBS.get(job_id, {})
                    job.update(status="cancelled", error="Generation cancelled before it started.")
                return
            time.sleep(0.5)
        # Rebuild upload streams for every attempt: a busy-GPU 429 consumes the
        # request body, so retrying with the same BytesIO objects would send empty files.
        # Reuse the established validation, pipeline and progress logic without
        # keeping the original Cloudflare HTTP request open during inference.
        while True:
            if _remote_generation_cancelled(job_id):
                with REMOTE_GENERATION_JOBS_LOCK:
                    job = REMOTE_GENERATION_JOBS.get(job_id, {})
                    job.update(status="cancelled", error="Generation cancelled before it started.")
                return
            payload = dict(form_data)
            for field, (filename, content, content_type) in file_data.items():
                payload[field] = (io.BytesIO(content), filename, content_type)
            with worker.test_request_context(
                "/api/remote/generate", method="POST", data=payload
            ):
                response = worker.make_response(remote_generate())
                response.direct_passthrough = False
                status_code = response.status_code
                body = response.get_data()
                content_type = response.mimetype
            if status_code == 429:
                # Another worker workload acquired the shared GPU lock first.
                # Keep this task queued instead of failing the face-angle batch.
                time.sleep(1.0)
                continue
            if status_code < 200 or status_code >= 300:
                try:
                    error_payload = json.loads(body.decode("utf-8", errors="replace"))
                    error = error_payload.get("error") or error_payload.get("message") or body[:2000].decode("utf-8", errors="replace")
                except (ValueError, AttributeError):
                    error = body[:2000].decode("utf-8", errors="replace")
                with REMOTE_GENERATION_JOBS_LOCK:
                    job = REMOTE_GENERATION_JOBS.get(job_id, {})
                    job.update(status="failed", error=str(error)[:2500])
                return
            with REMOTE_GENERATION_JOBS_LOCK:
                job = REMOTE_GENERATION_JOBS.get(job_id, {})
                job.update(status="completed", result=body, content_type=content_type, finished_at=time.time())
            return
    except Exception as exc:
        with REMOTE_GENERATION_JOBS_LOCK:
            job = REMOTE_GENERATION_JOBS.get(job_id, {})
            job.update(status="failed", error=str(exc)[:2500], finished_at=time.time())


@worker.post("/api/remote/generate-async")
def remote_generate_async():
    if not torch.cuda.is_available():
        return jsonify(error="This cloud GPU worker has no CUDA GPU attached."), 503
    task_id = (request.form.get("task_id") or "").strip().lower()
    if not re.fullmatch(r"[a-f0-9]{32}", task_id):
        return jsonify(error="Invalid generation task ID."), 400
    with REMOTE_GENERATION_JOBS_LOCK:
        existing = REMOTE_GENERATION_JOBS.get(task_id)
        if existing:
            return jsonify(job_id=task_id, status=existing.get("status", "queued")), 202
        if len(REMOTE_GENERATION_JOBS) >= 40:
            completed = [
                key for key, value in REMOTE_GENERATION_JOBS.items()
                if value.get("status") in {"completed", "failed", "cancelled"}
            ]
            for key in completed[:20]:
                REMOTE_GENERATION_JOBS.pop(key, None)
        form_data = request.form.to_dict(flat=True)
        file_data = {}
        for field, uploaded in request.files.items():
            file_data[field] = (
                Path(uploaded.filename or "upload.bin").name,
                uploaded.stream.read(),
                uploaded.mimetype or "application/octet-stream",
            )
        REMOTE_GENERATION_JOBS[task_id] = {
            "job_id": task_id, "task_id": task_id, "status": "queued",
            "created_at": time.time(), "error": None,
        }
    REMOTE_GENERATION_POOL.submit(_run_async_remote_generation, task_id, form_data, file_data)
    return jsonify(job_id=task_id, status="queued"), 202


@worker.get("/api/remote/generation-jobs/<job_id>")
def remote_generation_job_status(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id or ""):
        return jsonify(error="Invalid generation job ID."), 400
    with REMOTE_GENERATION_JOBS_LOCK:
        job = REMOTE_GENERATION_JOBS.get(job_id)
        if not job:
            return jsonify(error="Generation job not found; the worker may have restarted."), 404
        payload = {key: value for key, value in job.items() if key != "result"}
    with GENERATION_PROGRESS_LOCK:
        progress = GENERATION_PROGRESS.get(job_id)
        if progress:
            payload["progress"] = dict(progress)
    return jsonify(payload)


@worker.get("/api/remote/generation-jobs/<job_id>/result")
def remote_generation_job_result(job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id or ""):
        return jsonify(error="Invalid generation job ID."), 400
    with REMOTE_GENERATION_JOBS_LOCK:
        job = REMOTE_GENERATION_JOBS.get(job_id)
        if not job:
            return jsonify(error="Generation job not found; the worker may have restarted."), 404
        if job.get("status") != "completed" or not job.get("result"):
            return jsonify(error="Generation result is not ready."), 409
        result = job["result"]
        content_type = job.get("content_type") or "image/png"
    return send_file(io.BytesIO(result), mimetype=content_type, download_name="generated.png")


@worker.post("/api/remote/generate")
def remote_generate():
    prompt = (request.form.get("prompt") or "").strip()
    negative_prompt = (request.form.get("negative_prompt") or "").strip()
    try:
        steps = int(request.form.get("steps", "25"))
        strength = float(request.form.get("strength", "0.45"))
        aspect_ratio = (request.form.get("aspect_ratio") or "portrait").strip().lower()
        image_preset = validate_image_preset(request.form.get("image_preset") or "full_body_photo")
        width, height = resolve_image_dimensions(aspect_ratio)
        profile_id = validate_profile_id(request.form["face_profile_id"]) if request.form.get("face_profile_id") else None
        seed = resolve_generation_seed(request.form.get("seed"))
        use_global_body_model = str(request.form.get("use_global_body_model") or "").strip().lower() in {"1", "true", "yes", "on"}
        face_angle = (request.form.get("face_angle") or "").strip().lower()
        face_expression = (request.form.get("face_expression") or "").strip().lower()
        pose_control_type = (request.form.get("pose_control_type") or "face_mesh").strip().lower()
        if pose_control_type not in {"face_mesh", "body_openpose"}:
            raise ValueError("Unsupported pose control type.")
        if face_angle not in {"", "front", "three_quarter_left", "three_quarter_right", "left_profile", "right_profile"}:
            raise ValueError("Unsupported face angle.")
        if face_expression not in {"", "neutral", "smile", "laugh", "serious", "surprised"}:
            raise ValueError("Unsupported facial expression.")
        task_id = (request.form.get("task_id") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid generation task ID.")
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc) or "Invalid generation settings."), 400
    if not prompt or len(prompt) > 1200 or len(negative_prompt) > 600:
        return jsonify(error="Prompt is required and must be within the allowed length."), 400
    if not 5 <= steps <= 25 or not 0.2 <= strength <= 0.95:
        return jsonify(error="Invalid step count or image strength."), 400
    if not torch.cuda.is_available():
        _set_generation_progress(task_id, asset_type="image", status="failed", progress=0, total_steps=steps, stage="failed", message="CUDA is not available on this worker.")
        return jsonify(error="CUDA is not available on this worker."), 503
    if profile_id:
        profile = get_profile(profile_id)
        if not profile or not profile.get("local_model_ready"):
            return jsonify(error="The selected face LoRA is not trained on this GPU worker."), 400
    if use_global_body_model:
        body_profile = local_engine.ensure_global_body_profile()
        if not local_engine.profile_ready_for_backend(local_engine.GLOBAL_BODY_PROFILE_ID, "local"):
            return jsonify(error="The shared body-type LoRA is not trained on this cloud GPU worker. Train it using the global body model first."), 400
        prompt = local_engine.GLOBAL_BODY_PROFILE_TOKEN + ", " + prompt
    image = None
    pose_control_image = None
    uploaded = request.files.get("image")
    if uploaded and uploaded.filename:
        try:
            with Image.open(uploaded.stream) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
            image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="Reference image is not a valid PNG, JPG or WEBP image."), 400
    pose_uploaded = request.files.get("pose_control_image")
    if pose_uploaded and pose_uploaded.filename:
        try:
            with Image.open(pose_uploaded.stream) as source:
                pose_control_image = ImageOps.exif_transpose(source).convert("RGB")
                pose_control_image.load()
            pose_control_image = ImageOps.fit(pose_control_image, (width, height), method=Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="Face-pose control image is not a valid image."), 400
        if image is None and pose_control_type != "body_openpose":
            return jsonify(error="Face-mesh pose control requires a source reference image."), 400
    _set_generation_progress(task_id, asset_type="image", status="running", progress=2, total_steps=steps, stage="loading_model", message="Loading the image model and selected face on the cloud GPU…")
    if not local_engine.generation_lock.acquire(blocking=False):
        _set_generation_progress(task_id, asset_type="image", status="failed", progress=0, total_steps=steps, stage="busy", message="Another GPU generation is already running.")
        return jsonify(error="Another GPU generation is running. Try again shortly."), 429
    try:
        mode = (
            ("body_pose_img2img" if image is not None else "body_pose_txt2img")
            if pose_control_image is not None and pose_control_type == "body_openpose"
            else "pose_img2img" if pose_control_image is not None
            else ("img2img" if image is not None else "txt2img")
        )
        pipeline = local_engine._load_cpu_or_gpu_pipeline(
            mode, profile_id, "local_gpu", use_global_body_model=use_global_body_model,
            face_angle=face_angle, face_expression=face_expression,
        )
        _set_generation_progress(task_id, asset_type="image", status="running", progress=5, total_steps=steps, stage="denoising", message="Starting image denoising steps on the cloud GPU…")
        def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
            if _remote_generation_cancelled(task_id):
                raise RemoteGenerationCancelled("Generation cancelled by the user.")
            scheduler_steps = len(getattr(active_pipeline.scheduler, "timesteps", [])) or steps
            current = min(step_index + 1, scheduler_steps)
            percent = min(92, 5 + round(87 * current / max(scheduler_steps, 1)))
            _set_generation_progress(
                task_id, asset_type="image", status="running", progress=percent,
                current_step=current, total_steps=scheduler_steps, stage="denoising",
                message=f"Generating image on the cloud GPU · step {current} of {scheduler_steps}",
            )
            return callback_kwargs
        generator = torch.Generator(device="cuda").manual_seed(seed)
        if str(request.form.get("prompt_is_final") or "").strip().lower() in {"1", "true", "yes", "on"}:
            effective_prompt, effective_negative_prompt = prompt, negative_prompt
        else:
            effective_prompt, effective_negative_prompt = build_generation_prompts(
                prompt, negative_prompt, image_preset, has_reference_image=image is not None
            )
        kwargs = {
            "prompt": effective_prompt,
            "negative_prompt": effective_negative_prompt or None,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": 7.5,
            "generator": generator,
        }
        if image is not None:
            kwargs.update(image=image, strength=strength)
            if getattr(pipeline, "_private_face_ip_adapter_loaded", False):
                kwargs["ip_adapter_image"] = image
        if pose_control_image is not None:
            kwargs["control_image"] = pose_control_image
            scale_key = "BODY_POSE_CONTROLNET_SCALE" if pose_control_type == "body_openpose" else "FACE_POSE_CONTROLNET_SCALE"
            try:
                kwargs["controlnet_conditioning_scale"] = float(os.environ.get(scale_key, "1.0"))
            except ValueError:
                kwargs["controlnet_conditioning_scale"] = 1.0
        result = pipeline(**kwargs, callback_on_step_end=on_step_end).images[0]
        _set_generation_progress(task_id, asset_type="image", status="running", progress=95, current_step=steps, total_steps=steps, stage="finalizing", message="Preparing the generated image for download…")
        output = io.BytesIO()
        result.save(output, format="PNG")
        output.seek(0)
        _set_generation_progress(task_id, asset_type="image", status="completed", progress=100, current_step=steps, total_steps=steps, stage="completed", message="Image generation completed on the cloud GPU.")
        response = send_file(output, mimetype="image/png", download_name="generated.png")
        response.headers["X-Component-Adapter-Status"] = json.dumps(
            getattr(pipeline, "_private_component_adapter_status", {})
        )
        return response
    except RemoteGenerationCancelled:
        _set_generation_progress(
            task_id, asset_type="image", status="cancelled", progress=0, total_steps=steps,
            stage="cancelled", message="Generation cancelled by the user.",
        )
        return jsonify(status="cancelled", task_id=task_id, message="Generation cancelled."), 409
    except Exception as exc:
        _set_generation_progress(task_id, asset_type="image", status="failed", progress=0, total_steps=steps, stage="failed", message="Image generation failed on the cloud GPU.")
        message = str(exc)
        if "out of memory" in message.lower():
            message = "GPU out of memory. Try fewer steps or a smaller request."
        return jsonify(error=f"GPU image generation failed: {message}"), 500
    finally:
        local_engine.generation_lock.release()
        with GENERATION_CANCEL_LOCK:
            GENERATION_CANCEL_REQUESTS.discard(task_id)


@worker.post("/api/remote/generate-video")
def remote_generate_video():
    global VIDEO_PIPE, VIDEO_ACTIVE_LORA
    prompt = (request.form.get("prompt") or "").strip()
    negative_prompt = (request.form.get("negative_prompt") or "").strip()
    try:
        steps = int(request.form.get("steps", "12"))
        frames = int(request.form.get("frames", "16"))
        fps = int(request.form.get("fps", "8"))
        seed = int(request.form.get("seed", "0"))
        profile_id = validate_profile_id(request.form["face_profile_id"]) if request.form.get("face_profile_id") else None
        face_angle = (request.form.get("face_angle") or "").strip().lower()
        face_expression = (request.form.get("face_expression") or "").strip().lower()
        if face_angle not in {"", "front", "three_quarter_left", "three_quarter_right", "left_profile", "right_profile"}:
            raise ValueError("Unsupported video face angle.")
        if face_expression not in {"", "neutral", "smile", "laugh", "serious", "surprised"}:
            raise ValueError("Unsupported video expression.")
        video_settings = validate_video_settings(steps=steps, frames=frames, fps=fps, seed=seed)
        steps, frames, fps, seed = (
            video_settings["steps"], video_settings["frames"],
            video_settings["fps"], video_settings["seed"],
        )
        task_id = (request.form.get("task_id") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid generation task ID.")
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc) or "Invalid video settings."), 400
    if not prompt or len(prompt) > 1000 or len(negative_prompt) > 600:
        return jsonify(error="Video prompt is required and must be within the allowed length."), 400
    if not torch.cuda.is_available():
        _set_generation_progress(task_id, asset_type="video", status="failed", progress=0, total_steps=steps, stage="failed", message="CUDA is not available on this worker.")
        return jsonify(error="CUDA is not available on this worker."), 503
    if not (local_engine.MOTION_DIR / "config.json").is_file():
        _set_generation_progress(task_id, asset_type="video", status="failed", progress=0, total_steps=steps, stage="failed", message="AnimateDiff motion adapter is missing on the GPU worker.")
        return jsonify(error="AnimateDiff motion adapter is missing on the GPU worker."), 400
    if profile_id:
        profile = get_profile(profile_id)
        if not profile or not profile.get("local_model_ready"):
            return jsonify(error="The selected face LoRA is not trained on this GPU worker."), 400
    _set_generation_progress(task_id, asset_type="video", status="running", progress=2, total_steps=steps, stage="loading_model", message="Loading the video model and selected face on the cloud GPU…")
    if not local_engine.generation_lock.acquire(blocking=False):
        _set_generation_progress(task_id, asset_type="video", status="failed", progress=0, total_steps=steps, stage="busy", message="Another GPU generation is already running.")
        return jsonify(error="Another GPU generation is running. Try again shortly."), 429
    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter
        from diffusers.utils import export_to_video
        with VIDEO_LOCK:
            if VIDEO_PIPE is None:
                adapter = MotionAdapter.from_pretrained(
                    str(local_engine.MOTION_DIR), local_files_only=True, torch_dtype=torch.float16
                )
                VIDEO_PIPE = AnimateDiffPipeline.from_pretrained(
                    str(local_engine.MODEL_DIR), motion_adapter=adapter, local_files_only=True,
                    torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
                )
                VIDEO_PIPE.enable_vae_slicing()
                try:
                    VIDEO_PIPE.enable_attention_slicing("max")
                except Exception:
                    pass
                VIDEO_PIPE.to("cuda")
                VIDEO_ACTIVE_LORA = None
            requested_adapters = (profile_id, face_angle, face_expression)
            if VIDEO_ACTIVE_LORA != requested_adapters:
                if VIDEO_ACTIVE_LORA is not None:
                    VIDEO_PIPE.unload_lora_weights()
                adapter_names = []
                adapter_weights = []
                component_status = {"angle": "not_requested", "expression": "not_requested"}
                if profile_id:
                    VIDEO_PIPE.load_lora_weights(str(model_path(profile_id)), adapter_name="face_identity")
                    adapter_names.append("face_identity")
                    adapter_weights.append(1.0)
                    for component_kind, selected_value in (("angle", face_angle), ("expression", face_expression)):
                        if not selected_value:
                            continue
                        component_path = model_path(profile_id) / "components" / component_kind
                        component_weights = (component_path / "pytorch_lora_weights.safetensors", component_path / "pytorch_lora_weights.bin")
                        if any(path.is_file() for path in component_weights):
                            adapter_name = "face_" + component_kind
                            VIDEO_PIPE.load_lora_weights(str(component_path), adapter_name=adapter_name)
                            adapter_names.append(adapter_name)
                            env_key = "FACE_ANGLE_LORA_SCALE" if component_kind == "angle" else "FACE_EXPRESSION_LORA_SCALE"
                            default_weight = "0.45" if component_kind == "angle" else "0.55"
                            try:
                                weight = float(os.environ.get(env_key, default_weight))
                            except ValueError:
                                weight = float(default_weight)
                            adapter_weights.append(max(0.0, min(1.0, weight)))
                            component_status[component_kind] = "loaded"
                        else:
                            component_status[component_kind] = "not_trained_prompt_only"
                if adapter_names:
                    VIDEO_PIPE.set_adapters(adapter_names, adapter_weights=adapter_weights)
                VIDEO_PIPE._private_video_component_adapter_status = component_status
                VIDEO_ACTIVE_LORA = requested_adapters
        _set_generation_progress(task_id, asset_type="video", status="running", progress=5, total_steps=steps, stage="denoising", message="Starting video denoising steps on the cloud GPU…")
        def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
            scheduler_steps = len(getattr(active_pipeline.scheduler, "timesteps", [])) or steps
            current = min(step_index + 1, scheduler_steps)
            percent = min(92, 5 + round(87 * current / max(scheduler_steps, 1)))
            _set_generation_progress(
                task_id, asset_type="video", status="running", progress=percent,
                current_step=current, total_steps=scheduler_steps, stage="denoising",
                message=f"Generating video on the cloud GPU · step {current} of {scheduler_steps} · {frames} frames",
            )
            return callback_kwargs
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = VIDEO_PIPE(
            prompt=prompt, negative_prompt=negative_prompt or None,
            num_frames=frames, num_inference_steps=steps,
            guidance_scale=7.0, generator=generator,
            callback_on_step_end=on_step_end,
        )
        _set_generation_progress(task_id, asset_type="video", status="running", progress=95, current_step=steps, total_steps=steps, stage="encoding", message="Encoding generated frames into MP4…")
        filename = f"gpu_{uuid.uuid4().hex}.mp4"
        video_path = local_engine.VIDEO_OUTPUT_DIR / filename
        export_to_video(result.frames[0], str(video_path), fps=fps)
        _set_generation_progress(task_id, asset_type="video", status="completed", progress=100, current_step=steps, total_steps=steps, stage="completed", message="Video generation completed on the cloud GPU.")
        response = send_file(video_path, mimetype="video/mp4", download_name="generated.mp4")
        response.headers["X-Component-Adapter-Status"] = json.dumps(
            getattr(VIDEO_PIPE, "_private_video_component_adapter_status", {})
        )
        return response
    except Exception as exc:
        _set_generation_progress(task_id, asset_type="video", status="failed", progress=0, total_steps=steps, stage="failed", message="Video generation failed on the cloud GPU.")
        message = str(exc)
        if "out of memory" in message.lower():
            message = "GPU out of memory. Try fewer frames."
        return jsonify(error=f"GPU video generation failed: {message}"), 500
    finally:
        local_engine.generation_lock.release()


if __name__ == "__main__":
    if not os.environ.get("REMOTE_API_TOKEN", "").strip():
        raise SystemExit("Set REMOTE_API_TOKEN to a strong secret before starting the GPU worker.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Start a CUDA-enabled cloud GPU Pod and verify PyTorch sees the GPU.")
    print(f"cloud GPU worker starting on port {int(os.environ.get('GPU_WORKER_PORT', '8000'))}")
    worker.run(
        host="0.0.0.0",
        port=int(os.environ.get("GPU_WORKER_PORT", "8000")),
        debug=False,
        threaded=True,
    )
