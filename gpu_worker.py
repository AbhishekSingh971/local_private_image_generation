"""Provider-neutral authenticated worker for a user-managed remote CUDA GPU.

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
from image_upscaling import resolve_upscale_options, upscale_image
from generation_contracts import preserve_generation_text
from model_registry import model_configuration, resolve_model_family, supports_face_lora
from face_profiles import (
    create_profile,
    dataset_dir,
    dataset_images,
    get_profile,
    model_path,
    update_profile,
    validate_profile_id,
)

worker = Flask("remote_gpu_worker")
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


def _training_job(profile_id, job_id, steps, resolution, approved_filenames):
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
    ]
    for filename in approved_filenames:
        command.extend(["--approved-filename", filename])
    with TASKS_LOCK:
        TASKS[job_id] = {
            "job_id": job_id, "profile_id": profile_id, "status": "running",
            "started_at": time.time(), "logs": "",
        }
    try:
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=8 * 60 * 60,
            check=False,
        )
        logs = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        if completed.returncode == 0 and (
            (model_path(profile_id) / "pytorch_lora_weights.safetensors").is_file()
            or (model_path(profile_id) / "pytorch_lora_weights.bin").is_file()
        ):
            update_profile(
                profile_id, local_model_ready=True, local_training_state="completed",
                local_training_error=None,
            )
            task = {"job_id": job_id, "profile_id": profile_id, "status": "completed", "logs": logs[-5000:]}
        else:
            error = logs[-3500:] or f"Training exited with code {completed.returncode}."
            update_profile(
                profile_id, local_training_state="failed", local_training_error=error,
                local_model_ready=False,
            )
            task = {"job_id": job_id, "profile_id": profile_id, "status": "failed", "error": error, "logs": logs[-5000:]}
    except Exception as exc:
        error = str(exc)
        update_profile(
            profile_id, local_training_state="failed", local_training_error=error,
            local_model_ready=False,
        )
        task = {"job_id": job_id, "profile_id": profile_id, "status": "failed", "error": error}
    cleanup_warnings = []
    for filename in approved_filenames:
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.startswith("approved_"):
            continue
        image_path = dataset_dir(profile_id) / safe_name
        try:
            image_path.unlink(missing_ok=True)
            image_path.with_name(image_path.name + ".caption.txt").unlink(missing_ok=True)
        except OSError as cleanup_error:
            cleanup_warnings.append(f"Could not remove temporary training photo {safe_name}: {cleanup_error}")
    if cleanup_warnings:
        task["cleanup_warnings"] = cleanup_warnings[:10]
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
        existing = get_profile(profile_id)
        if existing and existing.get("local_training_state") == "running":
            return jsonify(error="This face profile is already training on the GPU worker."), 409
        create_profile(name, profile_id=profile_id, token=token, description=description)
        approved_filenames = []
        worker_dataset = dataset_dir(profile_id)
        for upload, image in zip(photo_uploads, images):
            filename = f"approved_{uuid.uuid4().hex}.jpg"
            saved_path = worker_dataset / filename
            image.convert("RGB").save(saved_path, format="JPEG", quality=95, optimize=True)
            caption = caption_by_name.get(Path(upload.filename or "").name, "")
            caption_path = saved_path.with_name(saved_path.name + ".caption.txt")
            if caption:
                caption_path.write_text(caption[:1200], encoding="utf-8")
            else:
                caption_path.unlink(missing_ok=True)
            approved_filenames.append(saved_path.name)
        job_id = uuid.uuid4().hex
        update_profile(
            profile_id, local_training_state="running", local_training_job_id=job_id,
            local_training_error=None, local_model_ready=False,
        )
        with TASKS_LOCK:
            TASKS[job_id] = {"job_id": job_id, "profile_id": profile_id, "status": "queued"}
        TRAIN_POOL.submit(_training_job, profile_id, job_id, steps, resolution, approved_filenames)
        return jsonify(status="queued", job_id=job_id, profile_id=profile_id), 202
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
    prompt = request.form.get("prompt") or ""
    negative_prompt = request.form.get("negative_prompt") or ""
    try:
        steps = int(request.form.get("steps", "4" if (request.form.get("model_family") or "sd15").strip().lower() == "flux2_klein" else "25"))
        strength = float(request.form.get("strength", "0.45"))
        aspect_ratio = (request.form.get("aspect_ratio") or "portrait").strip().lower()
        image_preset = validate_image_preset(request.form.get("image_preset") or "full_body_photo")
        model_family = resolve_model_family(request.form.get("model_family") or "sd15")
        width, height = resolve_image_dimensions(aspect_ratio, model_family)
        profile_id = validate_profile_id(request.form["face_profile_id"]) if request.form.get("face_profile_id") else None
        seed = resolve_generation_seed(request.form.get("seed"))
        use_global_body_model = str(request.form.get("use_global_body_model") or "").strip().lower() in {"1", "true", "yes", "on"}
        upscale_options = resolve_upscale_options(
            request.form.get("ai_upscale"), request.form.get("upscale_steps") or 20
        )
        ai_upscale = upscale_options["enabled"]
        task_id = (request.form.get("task_id") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid generation task ID.")
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc) or "Invalid generation settings."), 400
    if not prompt.strip():
        return jsonify(error="Prompt is required."), 400
    if model_family == "flux2_klein":
        if negative_prompt.strip():
            return jsonify(error="FLUX.2 Klein 4B distilled does not accept a negative prompt."), 400
        if steps != 4:
            return jsonify(error="FLUX.2 Klein 4B distilled uses a fixed 4-step generation schedule."), 400
    elif not 5 <= steps <= 25 or not 0.2 <= strength <= 0.95:
        return jsonify(error="Invalid step count or image strength."), 400
    if not torch.cuda.is_available():
        _set_generation_progress(task_id, asset_type="image", status="failed", progress=0, total_steps=steps, stage="failed", message="CUDA is not available on this worker.")
        return jsonify(error="CUDA is not available on this worker."), 503
    if profile_id:
        if not supports_face_lora(model_family):
            return jsonify(error="This face LoRA was trained for SD 1.5 and cannot be loaded by the selected model family. Select SD 1.5 for this identity or continue without the trained face."), 400
        profile = get_profile(profile_id)
        if not profile or not profile.get("local_model_ready"):
            return jsonify(error="The selected face LoRA is not trained on this GPU worker."), 400
    if use_global_body_model:
        if model_family != "sd15":
            use_global_body_model = False
        else:
            try:
                local_engine.ensure_global_body_profile()
                if not local_engine.profile_ready_for_backend(local_engine.GLOBAL_BODY_PROFILE_ID, "local"):
                    use_global_body_model = False
                else:
                    prompt = local_engine.GLOBAL_BODY_PROFILE_TOKEN + ", " + prompt
            except (RuntimeError, ValueError):
                use_global_body_model = False
    image = None
    pose_control_image = None
    uploaded = request.files.get("image")
    if uploaded and uploaded.filename:
        try:
            with Image.open(uploaded.stream) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
            if model_family != "flux2_klein":
                image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="Reference image is not a valid PNG, JPG or WEBP image."), 400
    pose_uploaded = request.files.get("pose_control_image")
    if pose_uploaded and pose_uploaded.filename:
        if model_family != "sd15":
            return jsonify(error="The configured face-pose ControlNet is SD 1.5-only. Select SD 1.5 for pose-guided generation."), 400
        try:
            with Image.open(pose_uploaded.stream) as source:
                pose_control_image = ImageOps.exif_transpose(source).convert("RGB")
                pose_control_image.load()
            pose_control_image = ImageOps.fit(pose_control_image, (width, height), method=Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="Face-pose control image is not a valid image."), 400
        if image is None:
            return jsonify(error="Pose control requires a source reference image."), 400
    _set_generation_progress(task_id, asset_type="image", status="running", progress=2, total_steps=steps, stage="loading_model", message="Loading the image model and selected face on the cloud GPU…")
    if not local_engine.generation_lock.acquire(blocking=False):
        _set_generation_progress(task_id, asset_type="image", status="failed", progress=0, total_steps=steps, stage="busy", message="Another GPU generation is already running.")
        return jsonify(error="Another GPU generation is running. Try again shortly."), 429
    try:
        upsampled_prompt = ""
        prompt_token_count = None
        if model_family == "flux2_klein":
            if pose_control_image is not None:
                return jsonify(error="FLUX.2 Klein uses reference-image editing rather than the SD 1.5 face-pose ControlNet path."), 400
            if _remote_generation_cancelled(task_id):
                raise RemoteGenerationCancelled("Generation cancelled by the user.")
            from flux2_engine import generate_flux2_image
            _set_generation_progress(
                task_id, asset_type="image", status="running", progress=5,
                total_steps=steps, stage="loading_model",
                message="Loading FLUX.2 Klein 4B on the remote GPU…",
            )
            flux_output = generate_flux2_image(
                prompt, image=image, seed=seed,
                width=width, height=height,
                source=model_configuration("flux2_klein")["source"],
            )
            result = flux_output["image"]
            prompt_token_count = flux_output.get("prompt_token_count")
            if _remote_generation_cancelled(task_id):
                raise RemoteGenerationCancelled("Generation cancellation was requested; FLUX.2 inference has completed.")
        else:
            mode = "pose_img2img" if pose_control_image is not None else ("img2img" if image is not None else "txt2img")
            pipeline = local_engine._load_cpu_or_gpu_pipeline(
                mode, profile_id, "local_gpu", use_global_body_model=use_global_body_model,
                model_family=model_family,
            )
            _set_generation_progress(task_id, asset_type="image", status="running", progress=5, total_steps=steps, stage="denoising", message="Starting image denoising steps on the remote GPU…")
            def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
                if _remote_generation_cancelled(task_id):
                    raise RemoteGenerationCancelled("Generation cancelled by the user.")
                scheduler_steps = len(getattr(active_pipeline.scheduler, "timesteps", [])) or steps
                current = min(step_index + 1, scheduler_steps)
                percent = min(92, 5 + round(87 * current / max(scheduler_steps, 1)))
                _set_generation_progress(
                    task_id, asset_type="image", status="running", progress=percent,
                    current_step=current, total_steps=scheduler_steps, stage="denoising",
                    message=f"Generating image on the remote GPU · step {current} of {scheduler_steps}",
                )
                return callback_kwargs
            generator = torch.Generator(device="cuda").manual_seed(seed)
            if str(request.form.get("prompt_is_final") or "").strip().lower() in {"1", "true", "yes", "on"}:
                effective_prompt, effective_negative_prompt = prompt, negative_prompt
            else:
                effective_prompt, effective_negative_prompt = build_generation_prompts(
                    prompt, negative_prompt, image_preset, has_reference_image=image is not None
                )
            from generation_contracts import validate_diffusion_prompt_capacity
            prompt_token_count = validate_diffusion_prompt_capacity(
                pipeline, effective_prompt, effective_negative_prompt or ""
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
            if pose_control_image is not None:
                kwargs["control_image"] = pose_control_image
                kwargs["controlnet_conditioning_scale"] = 1.0
                if getattr(pipeline, "_private_face_ip_adapter_loaded", False):
                    kwargs["ip_adapter_image"] = image
            result = pipeline(**kwargs, callback_on_step_end=on_step_end).images[0]
        upscale_warning = ""
        if ai_upscale:
            _set_generation_progress(
                task_id, asset_type="image", status="running", progress=93,
                current_step=0, total_steps=upscale_options["steps"], stage="upscaling",
                message="Running optional AI x4 super-resolution on the remote GPU…",
            )
            def on_upscale_step(current, total):
                percent = min(94, 93 + round(current / max(total, 1)))
                _set_generation_progress(
                    task_id, asset_type="image", status="running", progress=percent,
                    current_step=current, total_steps=total, stage="upscaling",
                    message=f"AI x4 super-resolution · step {current} of {total}",
                )
            try:
                result = upscale_image(
                    result, prompt, negative_prompt=negative_prompt, seed=seed,
                    steps=upscale_options["steps"],
                    cancelled=lambda: _remote_generation_cancelled(task_id),
                    progress_callback=on_upscale_step,
                )
            except RemoteGenerationCancelled:
                raise
            except Exception as upscale_exc:
                worker.logger.exception("AI upscaling failed; returning the base generated image")
                upscale_warning = "AI upscaling failed; the original generated image was returned. " + str(upscale_exc)[:300]
        _set_generation_progress(task_id, asset_type="image", status="running", progress=95, current_step=steps, total_steps=steps, stage="finalizing", message="Preparing the generated image for download…")
        output = io.BytesIO()
        result.save(output, format="PNG")
        output.seek(0)
        _set_generation_progress(task_id, asset_type="image", status="completed", progress=100, current_step=steps, total_steps=steps, stage="completed", message="Image generation completed on the cloud GPU.")
        response = send_file(output, mimetype="image/png", download_name="generated.png")
        if upscale_warning:
            response.headers["X-Upscale-Warning"] = upscale_warning.replace("\\n", " ")[:500]
        response.headers["X-Upscale-Status"] = "completed" if ai_upscale and not upscale_warning else ("failed_fallback" if upscale_warning else "not_requested")
        response.headers["X-Prompt-Token-Count"] = json.dumps(prompt_token_count, separators=(",", ":"))
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
    try:
        prompt = preserve_generation_text(request.form.get("prompt"), "Prompt")
        negative_prompt = preserve_generation_text(request.form.get("negative_prompt"), "Negative prompt")
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    try:
        steps = int(request.form.get("steps", "12"))
        frames = int(request.form.get("frames", "16"))
        seed = resolve_generation_seed(request.form.get("seed"))
        profile_id = validate_profile_id(request.form["face_profile_id"]) if request.form.get("face_profile_id") else None
        task_id = (request.form.get("task_id") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid generation task ID.")
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc) or "Invalid video settings."), 400
    if not prompt.strip():
        return jsonify(error="Video prompt is required."), 400
    if not 5 <= steps <= 20 or frames not in (8, 16, 24):
        return jsonify(error="Choose 5–20 steps and 8, 16, or 24 frames."), 400
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
                VIDEO_PIPE.to("cuda")
                VIDEO_ACTIVE_LORA = None
            if VIDEO_ACTIVE_LORA != profile_id:
                if VIDEO_ACTIVE_LORA is not None:
                    VIDEO_PIPE.unload_lora_weights()
                    VIDEO_ACTIVE_LORA = None
                if profile_id:
                    VIDEO_PIPE.load_lora_weights(str(model_path(profile_id)))
                    VIDEO_ACTIVE_LORA = profile_id
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
        from generation_contracts import validate_diffusion_prompt_capacity
        validate_diffusion_prompt_capacity(VIDEO_PIPE, prompt, negative_prompt)
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
        export_to_video(result.frames[0], str(video_path), fps=8)
        _set_generation_progress(task_id, asset_type="video", status="completed", progress=100, current_step=steps, total_steps=steps, stage="completed", message="Video generation completed on the cloud GPU.")
        return send_file(video_path, mimetype="video/mp4", download_name="generated.mp4")
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
