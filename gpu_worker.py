"""Authenticated HTTP worker intended to run on a user-created RunPod GPU Pod.

Set REMOTE_API_TOKEN to a strong random secret. Expose port 8000 only over HTTPS
through RunPod's Pod proxy. This worker does not create, start, or stop Pods.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import torch
from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps, UnidentifiedImageError

import app as local_engine
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


def _training_job(profile_id, job_id, steps, resolution):
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
        return jsonify(error="This RunPod worker has no CUDA GPU attached."), 503
    try:
        profile_id = validate_profile_id(request.form.get("profile_id", ""))
        name = (request.form.get("name") or "").strip()
        token = (request.form.get("token") or "").strip()
        description = request.form.get("description") or ""
        steps = int(request.form.get("steps", "800"))
        resolution = int(request.form.get("resolution", "512"))
        if not 100 <= steps <= 3000 or resolution not in (384, 512):
            raise ValueError("Training steps must be 100–3000 and resolution must be 384 or 512.")
        images = local_engine._parse_uploads(request.files.getlist("photos"))
        if len(images) < 5:
            raise ValueError("At least 5 valid face photos are required.")
        create_profile(name, profile_id=profile_id, token=token, description=description)
        for image in images:
            save_profile_image(profile_id, image)
        existing = get_profile(profile_id)
        if existing and existing.get("local_training_state") == "running":
            return jsonify(error="This face profile is already training on the GPU worker."), 409
        job_id = uuid.uuid4().hex
        update_profile(
            profile_id, local_training_state="running", local_training_job_id=job_id,
            local_training_error=None, local_model_ready=False,
        )
        with TASKS_LOCK:
            TASKS[job_id] = {"job_id": job_id, "profile_id": profile_id, "status": "queued"}
        TRAIN_POOL.submit(_training_job, profile_id, job_id, steps, resolution)
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


@worker.post("/api/remote/generate")
def remote_generate():
    prompt = (request.form.get("prompt") or "").strip()
    negative_prompt = (request.form.get("negative_prompt") or "").strip()
    try:
        steps = int(request.form.get("steps", "10"))
        strength = float(request.form.get("strength", "0.45"))
        profile_id = validate_profile_id(request.form["face_profile_id"]) if request.form.get("face_profile_id") else None
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
    image = None
    uploaded = request.files.get("image")
    if uploaded and uploaded.filename:
        try:
            with Image.open(uploaded.stream) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
            image = ImageOps.fit(image, (512, 512), method=Image.Resampling.LANCZOS)
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            return jsonify(error="Reference image is not a valid PNG, JPG or WEBP image."), 400
    _set_generation_progress(task_id, asset_type="image", status="running", progress=2, total_steps=steps, stage="loading_model", message="Loading the image model and selected face on RunPod…")
    if not local_engine.generation_lock.acquire(blocking=False):
        _set_generation_progress(task_id, asset_type="image", status="failed", progress=0, total_steps=steps, stage="busy", message="Another GPU generation is already running.")
        return jsonify(error="Another GPU generation is running. Try again shortly."), 429
    try:
        mode = "img2img" if image is not None else "txt2img"
        pipeline = local_engine._load_cpu_or_gpu_pipeline(mode, profile_id, "local_gpu")
        _set_generation_progress(task_id, asset_type="image", status="running", progress=5, total_steps=steps, stage="denoising", message="Starting image denoising steps on RunPod…")
        def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
            scheduler_steps = len(getattr(active_pipeline.scheduler, "timesteps", [])) or steps
            current = min(step_index + 1, scheduler_steps)
            percent = min(92, 5 + round(87 * current / max(scheduler_steps, 1)))
            _set_generation_progress(
                task_id, asset_type="image", status="running", progress=percent,
                current_step=current, total_steps=scheduler_steps, stage="denoising",
                message=f"Generating image on RunPod · step {current} of {scheduler_steps}",
            )
            return callback_kwargs
        generator = torch.Generator(device="cuda").manual_seed(int(time.time_ns() % (2**31 - 1)))
        kwargs = {
            "prompt": prompt, "negative_prompt": negative_prompt or None,
            "num_inference_steps": steps, "guidance_scale": 7.0, "generator": generator,
        }
        if image is not None:
            kwargs.update(image=image, strength=strength)
        result = pipeline(**kwargs, callback_on_step_end=on_step_end).images[0]
        _set_generation_progress(task_id, asset_type="image", status="running", progress=95, current_step=steps, total_steps=steps, stage="finalizing", message="Preparing the generated image for download…")
        output = io.BytesIO()
        result.save(output, format="PNG")
        output.seek(0)
        _set_generation_progress(task_id, asset_type="image", status="completed", progress=100, current_step=steps, total_steps=steps, stage="completed", message="Image generation completed on RunPod.")
        return send_file(output, mimetype="image/png", download_name="generated.png")
    except Exception as exc:
        _set_generation_progress(task_id, asset_type="image", status="failed", progress=0, total_steps=steps, stage="failed", message="Image generation failed on RunPod.")
        message = str(exc)
        if "out of memory" in message.lower():
            message = "GPU out of memory. Try fewer steps or a smaller request."
        return jsonify(error=f"GPU image generation failed: {message}"), 500
    finally:
        local_engine.generation_lock.release()


@worker.post("/api/remote/generate-video")
def remote_generate_video():
    global VIDEO_PIPE, VIDEO_ACTIVE_LORA
    prompt = (request.form.get("prompt") or "").strip()
    negative_prompt = (request.form.get("negative_prompt") or "").strip()
    try:
        steps = int(request.form.get("steps", "12"))
        frames = int(request.form.get("frames", "16"))
        profile_id = validate_profile_id(request.form["face_profile_id"]) if request.form.get("face_profile_id") else None
        task_id = (request.form.get("task_id") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid generation task ID.")
    except (ValueError, TypeError, KeyError) as exc:
        return jsonify(error=str(exc) or "Invalid video settings."), 400
    if not prompt or len(prompt) > 1000 or len(negative_prompt) > 600:
        return jsonify(error="Video prompt is required and must be within the allowed length."), 400
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
    _set_generation_progress(task_id, asset_type="video", status="running", progress=2, total_steps=steps, stage="loading_model", message="Loading the video model and selected face on RunPod…")
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
        _set_generation_progress(task_id, asset_type="video", status="running", progress=5, total_steps=steps, stage="denoising", message="Starting video denoising steps on RunPod…")
        def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
            scheduler_steps = len(getattr(active_pipeline.scheduler, "timesteps", [])) or steps
            current = min(step_index + 1, scheduler_steps)
            percent = min(92, 5 + round(87 * current / max(scheduler_steps, 1)))
            _set_generation_progress(
                task_id, asset_type="video", status="running", progress=percent,
                current_step=current, total_steps=scheduler_steps, stage="denoising",
                message=f"Generating video on RunPod · step {current} of {scheduler_steps} · {frames} frames",
            )
            return callback_kwargs
        generator = torch.Generator(device="cuda").manual_seed(int(time.time_ns() % (2**31 - 1)))
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
        _set_generation_progress(task_id, asset_type="video", status="completed", progress=100, current_step=steps, total_steps=steps, stage="completed", message="Video generation completed on RunPod.")
        return send_file(video_path, mimetype="video/mp4", download_name="generated.mp4")
    except Exception as exc:
        _set_generation_progress(task_id, asset_type="video", status="failed", progress=0, total_steps=steps, stage="failed", message="Video generation failed on RunPod.")
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
        raise SystemExit("CUDA is not available. Start a CUDA-enabled RunPod Pod and verify PyTorch sees the GPU.")
    print(f"RunPod GPU worker starting on port {int(os.environ.get('GPU_WORKER_PORT', '8000'))}")
    worker.run(
        host="0.0.0.0",
        port=int(os.environ.get("GPU_WORKER_PORT", "8000")),
        debug=False,
        threaded=True,
    )
