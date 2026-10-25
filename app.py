import gc
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
from urllib.parse import urlparse

# Keep basic CPU inference conservative; model download telemetry is disabled.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from dotenv import load_dotenv
APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

import requests
import torch
from flask import Flask, request, send_from_directory, send_file, jsonify, render_template, redirect, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

from compute_routing import select_compute_backend
from face_profiles import (
    DATA_DIR,
    LORA_DIR,
    PROFILES_DIR,
    create_profile,
    delete_profile,
    dataset_images,
    dataset_dir,
    get_profile,
    list_profiles,
    model_path,
    profile_ready_for_backend,
    save_profile_image,
    update_profile,
    validate_profile_id,
)
from studio_db import (
    create_asset,
    database_status,
    database_url_configured,
    get_face_model_bundle,
    save_face_model_bundle,
    delete_asset,
    get_asset,
    list_assets,
    update_asset,
)

INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs"
VIDEO_OUTPUT_DIR = DATA_DIR / "videos"
MODEL_DIR = APP_DIR / "models" / "stable-diffusion-v1-5"
MOTION_DIR = APP_DIR / "models" / "animatediff-motion-adapter-v1-5-2"
for folder in (INPUT_DIR, OUTPUT_DIR, VIDEO_OUTPUT_DIR, MODEL_DIR, LORA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

Image.MAX_IMAGE_PIXELS = 40_000_000
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
MODEL_ID = os.environ.get("SD_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5")

pipe = None
pipeline_mode = None
pipeline_device = None
active_lora = None
model_error = None
video_pipe = None
active_video_lora = None
video_model_error = None
model_lock = threading.RLock()
generation_lock = threading.Lock()
TRAINING_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="face-lora-train")
LOCAL_TRAIN_JOBS = {}
LOCAL_TRAIN_JOBS_LOCK = threading.Lock()


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def runpod_configured() -> bool:
    return bool(
        os.environ.get("RUNPOD_WORKER_URL", "").strip()
        and os.environ.get("RUNPOD_WORKER_TOKEN", "").strip()
    )


def _runpod_url(path: str) -> str:
    base_url = os.environ.get("RUNPOD_WORKER_URL", "").strip().rstrip("/")
    token = os.environ.get("RUNPOD_WORKER_TOKEN", "").strip()
    if not base_url or not token:
        raise RuntimeError(
            "RunPod is not configured. Set RUNPOD_WORKER_URL and RUNPOD_WORKER_TOKEN in .env, "
            "then restart the app. The GPU toggle does not create or purchase a Pod."
        )
    parsed = urlparse(base_url)
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RuntimeError("The RunPod worker URL must use HTTPS.")
    return f"{base_url}{path}"


def runpod_request(method: str, path: str, *, data=None, files=None, timeout=(15, 90)):
    url = _runpod_url(path)
    token = os.environ.get("RUNPOD_WORKER_TOKEN", "").strip()
    try:
        response = requests.request(
            method, url, headers={"Authorization": f"Bearer {token}"},
            data=data, files=files, timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not reach the configured RunPod worker at {urlparse(url).netloc}: {exc}"
        ) from exc
    if not response.ok:
        try:
            detail = response.json().get("error") or response.json().get("message") or response.text
        except (ValueError, AttributeError):
            detail = response.text
        raise RuntimeError(f"RunPod worker returned HTTP {response.status_code}: {detail[:1200]}")
    return response


def _local_gpu_details() -> dict:
    """Report GPUs actually usable by this PyTorch installation (CUDA)."""
    details = {"available": False, "name": None, "vram_gb": None}
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            details.update({
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "vram_gb": round(props.total_memory / (1024 ** 3), 2),
            })
    except Exception as exc:
        details["error"] = str(exc)
    return details


def _compute_options(probe_cloud: bool = False) -> dict:
    local = _local_gpu_details()
    configured = runpod_configured()
    cloud = {"available": False, "name": None, "vram_gb": None}
    cloud_error = None
    if probe_cloud and configured:
        try:
            response = runpod_request("GET", "/api/remote/health", timeout=(8, 12))
            payload = response.json()
            if payload.get("cuda_available") is True:
                cloud.update({
                    "available": True,
                    "name": payload.get("gpu_name") or "RunPod GPU",
                    "vram_gb": payload.get("vram_gb"),
                })
            else:
                cloud_error = payload.get("error") or "RunPod worker reports no CUDA GPU."
        except (RuntimeError, ValueError) as exc:
            cloud_error = str(exc)

    available = ["cpu"]
    if local["available"]:
        available.append("local_gpu")
    if cloud["available"]:
        available.append("runpod")
    auto_backend = "local_gpu" if local["available"] else ("runpod" if cloud["available"] else "cpu")
    return {
        "local_gpu": local,
        "cloud_gpu": cloud,
        "runpod_configured": configured,
        "available_backends": available,
        "auto_backend": auto_backend,
        "cloud_error": cloud_error,
        "message": (
            f"Using local GPU: {local['name']}."
            if auto_backend == "local_gpu" else
            f"Local CUDA GPU not detected; cloud GPU available: {cloud['name']}."
            if auto_backend == "runpod" else
            "No usable GPU detected. CPU is available; configure a running RunPod GPU for acceleration."
        ),
    }


def _resolve_compute_backend(form) -> str:
    """OFF => CPU. ON => selected GPU; auto prefers local CUDA, then healthy RunPod."""
    gpu_enabled = _truthy(form.get("use_gpu", form.get("use_runpod")))
    if not gpu_enabled:
        return "cpu"
    requested = (form.get("compute_backend") or "auto").strip().lower()
    local_available = bool(torch.cuda.is_available())
    probe_cloud = requested == "runpod" or (requested == "auto" and not local_available)
    options = _compute_options(probe_cloud=probe_cloud)
    return select_compute_backend(
        True, requested,
        local_gpu_available=options["local_gpu"]["available"],
        cloud_gpu_available=options["cloud_gpu"]["available"],
    )


def _clean_prompt(value: str, field_name: str, limit: int) -> str:
    value = (value or "").strip()
    if len(value) > limit:
        raise ValueError(f"{field_name} must be {limit} characters or fewer.")
    return value


def _parse_uploads(file_storages, *, max_files=30):
    selected = [item for item in file_storages if item and item.filename]
    if len(selected) > max_files:
        raise ValueError(f"Upload at most {max_files} photos at a time.")
    prepared = []
    for item in selected:
        raw = item.stream.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Each photo must be 8 MB or smaller.")
        try:
            with Image.open(io.BytesIO(raw)) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
            raise ValueError("One file is not a valid PNG, JPG or WEBP image.")
        if image.width < 128 or image.height < 128:
            raise ValueError("Each photo must be at least 128 × 128 pixels.")
        prepared.append(image.copy())
    return prepared


def _save_generated_png(image) -> str:
    filename = f"{uuid.uuid4().hex}.png"
    image.save(OUTPUT_DIR / filename, format="PNG", optimize=True)
    return filename


def _persist_generated_asset(asset_type, filename, *, title, prompt, negative_prompt, profile, backend, generation_mode):
    folder = OUTPUT_DIR if asset_type == "image" else VIDEO_OUTPUT_DIR
    path = folder / Path(filename).name
    try:
        return create_asset(
            asset_type=asset_type,
            title=title or prompt[:100] or ("Generated image" if asset_type == "image" else "Generated video"),
            filename=path.name,
            mime_type="image/png" if asset_type == "image" else "video/mp4",
            media_data=path.read_bytes(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            profile_id=profile["id"] if profile else None,
            profile_name=profile["name"] if profile else None,
            backend=backend,
            generation_mode=generation_mode,
        )
    except Exception as exc:
        app.logger.exception("Could not persist generated media to the database")
        raise RuntimeError(
            "The media was generated but could not be saved to the database. Check DATABASE_URL and database permissions. The local output copy remains in " + str(path)
        ) from exc


def _persist_face_model_bundle(profile_id: str, backend: str, bundle_data: bytes | None = None, file_count: int | None = None):
    """Back up a trained LoRA into PostgreSQL when DATABASE_URL is configured."""
    if not database_url_configured():
        return None
    if bundle_data is None:
        folder = model_path(profile_id)
        if not folder.is_dir():
            raise RuntimeError("Trained LoRA output folder is missing.")
        files = [item for item in folder.rglob("*") if item.is_file()]
        if not files:
            raise RuntimeError("No trained LoRA files were produced.")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for item in files:
                archive.write(item, arcname=item.relative_to(folder).as_posix())
        bundle_data = buffer.getvalue()
        file_count = len(files)
    return save_face_model_bundle(
        profile_id, backend, f"{profile_id}-{backend}-lora.zip",
        bundle_data, int(file_count or 1),
    )


def _run_local_training(profile_id: str, steps: int, resolution: int, job_id: str) -> None:
    profile = get_profile(profile_id)
    if not profile:
        return
    args = [
        sys.executable, str(APP_DIR / "train_face_lora.py"),
        "--profile-id", profile_id,
        "--output-name", profile_id,
        "--token", profile["token"],
        "--steps", str(steps),
        "--resolution", str(resolution),
        "--mixed-precision", "fp16",
    ]
    try:
        completed = subprocess.run(
            args, cwd=APP_DIR, capture_output=True, text=True, timeout=8 * 60 * 60,
            check=False,
        )
        logs = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if completed.returncode != 0:
            error = logs.strip()[-3500:] or f"Trainer exited with code {completed.returncode}."
            update_profile(
                profile_id, local_training_state="failed",
                local_training_error=error, local_model_ready=False,
            )
            with LOCAL_TRAIN_JOBS_LOCK:
                LOCAL_TRAIN_JOBS[job_id] = {"status": "failed", "error": error}
            return
        _persist_face_model_bundle(profile_id, "local")
        update_profile(
            profile_id, local_training_state="completed",
            local_training_error=None, local_model_ready=True,
        )
        with LOCAL_TRAIN_JOBS_LOCK:
            LOCAL_TRAIN_JOBS[job_id] = {"status": "completed", "message": "Face LoRA trained locally."}
    except Exception as exc:
        error = str(exc)
        update_profile(
            profile_id, local_training_state="failed",
            local_training_error=error, local_model_ready=False,
        )
        with LOCAL_TRAIN_JOBS_LOCK:
            LOCAL_TRAIN_JOBS[job_id] = {"status": "failed", "error": error}


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error="Upload too large. Keep the combined request under 64 MB."), 413


@app.context_processor
def inject_storage_status():
    return {"storage_status": database_status()}


@app.get("/")
def index():
    return redirect(url_for("dashboard"))


@app.get("/dashboard")
def dashboard():
    images = list_assets("image", 500)
    videos = list_assets("video", 500)
    return render_template(
        "dashboard.html", page_key="dashboard", active_page="dashboard",
        image_count=len(images), video_count=len(videos),
        profile_count=len(list_profiles()), recent_assets=list_assets(limit=8),
    )


@app.get("/generate/image")
def generate_image_page():
    return render_template("generate_image.html", page_key="image", active_page="image")


@app.get("/generate/video")
def generate_video_page():
    return render_template("generate_video.html", page_key="video", active_page="video")


@app.get("/faces")
def faces_page():
    return render_template("faces.html", page_key="faces", active_page="faces")


@app.get("/library/images")
def library_images_page():
    return render_template("library.html", page_key="images", active_page="images", asset_type="image")


@app.get("/library/videos")
def library_videos_page():
    return render_template("library.html", page_key="videos", active_page="videos", asset_type="video")


@app.get("/api/assets")
def api_list_assets():
    asset_type = (request.args.get("type") or "").strip().lower()
    if asset_type and asset_type not in {"image", "video"}:
        return jsonify(error="Asset type must be image or video."), 400
    try:
        limit = int(request.args.get("limit", "200"))
        assets = list_assets(asset_type or None, limit)
        for asset in assets:
            asset["content_url"] = url_for("asset_content", asset_id=asset["id"])
        return jsonify(assets=assets)
    except (TypeError, ValueError) as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Could not list media from the database")
        return jsonify(error="Could not read the media library. Check DATABASE_URL and database connectivity."), 503


@app.get("/assets/<asset_id>/content")
def asset_content(asset_id):
    if not re.fullmatch(r"[a-f0-9]{32}", asset_id or ""):
        return jsonify(error="Invalid media ID."), 400
    try:
        asset = get_asset(asset_id, include_content=True)
    except Exception:
        app.logger.exception("Could not read media from the database")
        return jsonify(error="Could not read saved media. Check DATABASE_URL and database connectivity."), 503
    if not asset:
        return jsonify(error="Saved media not found."), 404
    return send_file(
        io.BytesIO(asset["media_data"]), mimetype=asset["mime_type"],
        as_attachment=_truthy(request.args.get("download")),
        download_name=Path(asset["filename"]).name, max_age=0,
    )


@app.route("/api/assets/<asset_id>", methods=["PATCH", "DELETE"])
def api_asset_detail(asset_id):
    if not re.fullmatch(r"[a-f0-9]{32}", asset_id or ""):
        return jsonify(error="Invalid media ID."), 400
    if request.method == "PATCH":
        payload = request.get_json(silent=True) or {}
        try:
            asset = update_asset(asset_id, payload)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception("Could not update media metadata")
            return jsonify(error="Could not update media metadata in the database."), 503
        if not asset:
            return jsonify(error="Saved media not found."), 404
        return jsonify(status="updated", asset=asset)
    try:
        asset = delete_asset(asset_id)
    except Exception:
        app.logger.exception("Could not delete media from database")
        return jsonify(error="Could not delete media from the database."), 503
    if not asset:
        return jsonify(error="Saved media not found."), 404
    folder = OUTPUT_DIR if asset["asset_type"] == "image" else VIDEO_OUTPUT_DIR
    local_copy = folder / Path(asset["filename"]).name
    try:
        if local_copy.is_file():
            local_copy.unlink()
    except OSError:
        app.logger.warning("Deleted database asset %s but could not delete its local copy", asset_id)
    return jsonify(status="deleted", id=asset_id)


@app.delete("/api/face-profiles/<profile_id>")
def api_delete_face_profile(profile_id):
    try:
        profile_id = validate_profile_id(profile_id)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    profile = get_profile(profile_id)
    if not profile:
        return jsonify(error="Face profile not found."), 404
    if profile.get("local_training_state") == "running" or profile.get("runpod_training_state") == "running":
        return jsonify(error="This face profile is training. Wait for the job to finish before deleting it."), 409
    try:
        deleted = delete_profile(profile_id)
    except Exception:
        app.logger.exception("Could not delete face profile")
        return jsonify(error="Could not delete the face profile. Check the database connection."), 503
    if not deleted:
        return jsonify(error="Face profile not found."), 404
    return jsonify(status="deleted", id=profile_id)


@app.post("/api/face-profiles/<profile_id>/photos/from-assets")
def api_add_generated_photos(profile_id):
    try:
        profile_id = validate_profile_id(profile_id)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    profile = get_profile(profile_id)
    if not profile:
        return jsonify(error="Face profile not found."), 404
    payload = request.get_json(silent=True) or {}
    asset_ids = payload.get("asset_ids")
    if not isinstance(asset_ids, list) or not asset_ids:
        return jsonify(error="Select one or more generated images."), 400
    if len(asset_ids) > 30:
        return jsonify(error="Add at most 30 generated images per request."), 400
    if len(set(map(str, asset_ids))) != len(asset_ids):
        return jsonify(error="The selection contains duplicate image IDs."), 400
    added = 0
    try:
        for asset_id in asset_ids:
            asset_id = str(asset_id)
            if not re.fullmatch(r"[a-f0-9]{32}", asset_id):
                return jsonify(error="One selected image ID is invalid."), 400
            asset = get_asset(asset_id, include_content=True)
            if not asset or asset["asset_type"] != "image":
                return jsonify(error="One selected item is not a saved image."), 400
            with Image.open(io.BytesIO(asset["media_data"])) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.load()
            if image.width < 128 or image.height < 128:
                return jsonify(error="Every selected image must be at least 128 × 128 pixels."), 400
            save_profile_image(profile_id, image)
            added += 1
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        return jsonify(error="A selected media item is not a valid image: " + str(exc)), 400
    except Exception:
        app.logger.exception("Could not copy selected generated images to face dataset")
        return jsonify(error="Could not save selected images to the database or local dataset."), 503
    return jsonify(status="saved", added=added, profile=get_profile(profile_id))



@app.get("/health")
def health():
    return jsonify(
        status="ok",
        storage=database_status(),
        image_model_loaded=pipe is not None,
        image_model_error=model_error,
        face_profiles=len(list_profiles()),
        video_model_loaded=video_pipe is not None,
        video_adapter_ready=(MOTION_DIR / "config.json").is_file(),
        runpod_configured=runpod_configured(),
        local_gpu=_local_gpu_details(),
        note="GPU toggle off means local CPU. When on, available GPU backends can be selected.",
    )


@app.get("/api/gpu-status")
def gpu_status():
    probe_cloud = _truthy(request.args.get("probe"))
    result = _compute_options(probe_cloud=probe_cloud)
    result["configured"] = result["runpod_configured"]
    result["connected"] = result["cloud_gpu"]["available"]
    result["worker"] = {
        "cuda_available": result["cloud_gpu"]["available"],
        "gpu_name": result["cloud_gpu"]["name"],
        "vram_gb": result["cloud_gpu"]["vram_gb"],
    }
    if result["cloud_error"]:
        result["message"] = result["cloud_error"]
    return jsonify(result)


@app.get("/api/face-profiles")
def api_face_profiles():
    return jsonify(profiles=list_profiles())


@app.post("/api/face-profiles")
def api_create_face_profile():
    try:
        name = _clean_prompt(request.form.get("name"), "Profile name", 80)
        token = (request.form.get("token") or "").strip() or None
        uploads = request.files.getlist("photos")
        images = _parse_uploads(uploads) if any(item and item.filename for item in uploads) else []
        profile = create_profile(name, token=token)
        for image in images:
            save_profile_image(profile["id"], image)
        profile = get_profile(profile["id"])
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 500
    return jsonify(status="created", profile=profile), 201


@app.post("/api/face-profiles/<profile_id>/photos")
def api_add_face_photos(profile_id):
    try:
        profile_id = validate_profile_id(profile_id)
        if not get_profile(profile_id):
            return jsonify(error="Face profile not found."), 404
        images = _parse_uploads(request.files.getlist("photos"))
        if not images:
            return jsonify(error="Choose one or more photos to upload."), 400
        for image in images:
            save_profile_image(profile_id, image)
        return jsonify(status="saved", profile=get_profile(profile_id), added=len(images))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/face-profiles/<profile_id>/train")
def api_train_face_profile(profile_id):
    try:
        profile_id = validate_profile_id(profile_id)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    profile = get_profile(profile_id)
    if not profile:
        return jsonify(error="Face profile not found."), 404
    if profile["image_count"] < 5:
        return jsonify(error=f"Add at least 5 clear photos to this face profile first. Currently: {profile['image_count']}. "), 400
    try:
        steps = int(request.form.get("steps", "800"))
        resolution = int(request.form.get("resolution", "512"))
    except (TypeError, ValueError):
        return jsonify(error="Invalid training steps or resolution."), 400
    if not 100 <= steps <= 3000 or resolution not in (384, 512):
        return jsonify(error="Training steps must be 100–3000 and resolution must be 384 or 512."), 400

    try:
        backend = _resolve_compute_backend(request.form)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if backend == "cpu":
        return jsonify(error=(
            "Face LoRA training needs a GPU. Turn on Enable GPU acceleration, then select an available "
            "System GPU or RunPod GPU. CPU fallback is not supported for this training workflow."
        )), 400

    if backend == "runpod":
        if not runpod_configured():
            return jsonify(error=(
                "RunPod is not configured. Set RUNPOD_WORKER_URL and RUNPOD_WORKER_TOKEN in .env, "
                "deploy the worker on your RTX 3090 Pod, then restart the local app."
            )), 400
        opened_files = []
        try:
            for path in dataset_images(profile_id):
                handle = path.open("rb")
                opened_files.append(handle)
            files = [
                ("photos", (path.name, handle, "image/jpeg"))
                for path, handle in zip(dataset_images(profile_id), opened_files)
            ]
            response = runpod_request(
                "POST", "/api/remote/train-profile",
                data={
                    "profile_id": profile_id,
                    "name": profile["name"],
                    "token": profile["token"],
                    "steps": str(steps),
                    "resolution": str(resolution),
                },
                files=files, timeout=(15, 90),
            )
            job = response.json()
        except (RuntimeError, ValueError) as exc:
            return jsonify(error=str(exc)), 502
        finally:
            for handle in opened_files:
                handle.close()
        job_id = job.get("job_id")
        if not job_id:
            return jsonify(error="RunPod worker did not return a training job ID."), 502
        update_profile(
            profile_id, runpod_training_state="running", runpod_training_job_id=job_id,
            runpod_training_error=None, runpod_model_ready=False,
        )
        return jsonify(status="running", backend="runpod", job_id=job_id, message="RunPod face training has started."), 202

    if not torch.cuda.is_available():
        return jsonify(error=(
            "System GPU was selected, but this Python environment cannot access a CUDA GPU. "
            "Choose RunPod GPU or install a compatible CUDA-enabled PyTorch build."
        )), 400

    job_id = uuid.uuid4().hex
    update_profile(
        profile_id, local_training_state="running", local_training_job_id=job_id,
        local_training_error=None, local_model_ready=False,
    )
    with LOCAL_TRAIN_JOBS_LOCK:
        LOCAL_TRAIN_JOBS[job_id] = {"status": "running", "profile_id": profile_id}
    TRAINING_POOL.submit(_run_local_training, profile_id, steps, resolution, job_id)
    return jsonify(status="running", backend="local", job_id=job_id, message="Local face training has started."), 202


@app.get("/api/face-profiles/<profile_id>/train-status")
def api_face_train_status(profile_id):
    try:
        profile_id = validate_profile_id(profile_id)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    profile = get_profile(profile_id)
    if not profile:
        return jsonify(error="Face profile not found."), 404
    backend = request.args.get("backend", "local").strip().lower()
    if backend == "local":
        job_id = profile.get("local_training_job_id")
        with LOCAL_TRAIN_JOBS_LOCK:
            local_job = LOCAL_TRAIN_JOBS.get(job_id, {}) if job_id else {}
        return jsonify(
            status=profile.get("local_training_state", "not_trained"),
            error=profile.get("local_training_error"),
            job=local_job,
            profile=get_profile(profile_id),
        )
    if backend != "runpod":
        return jsonify(error="Backend must be local or runpod."), 400
    job_id = profile.get("runpod_training_job_id")
    if not job_id:
        return jsonify(status=profile.get("runpod_training_state", "not_trained"), profile=profile)
    try:
        payload = runpod_request("GET", f"/api/remote/jobs/{job_id}", timeout=(10, 15)).json()
    except RuntimeError as exc:
        return jsonify(error=str(exc), status="running", profile=profile), 502
    state = payload.get("status", "running")
    if state == "completed":
        if database_url_configured() and get_face_model_bundle(profile_id, "runpod") is None:
            try:
                model_response = runpod_request(
                    "GET", f"/api/remote/models/{profile_id}", timeout=(15, 120)
                )
                file_count = int(model_response.headers.get("X-Model-File-Count", "1"))
                _persist_face_model_bundle(profile_id, "runpod", model_response.content, file_count)
            except Exception as exc:
                app.logger.warning("Could not back up RunPod LoRA bundle to PostgreSQL: %s", type(exc).__name__)
        update_profile(
            profile_id, runpod_training_state="completed", runpod_training_error=None,
            runpod_model_ready=True,
        )
    elif state == "failed":
        update_profile(
            profile_id, runpod_training_state="failed",
            runpod_training_error=(payload.get("error") or "RunPod training failed")[-3500:],
            runpod_model_ready=False,
        )
    updated_profile = get_profile(profile_id)
    return jsonify(status=state, error=payload.get("error"), job=payload, profile=updated_profile)


def _load_cpu_or_gpu_pipeline(mode, profile_id, compute_backend="cpu"):
    global pipe, pipeline_mode, pipeline_device, active_lora, model_error
    global video_pipe, active_video_lora
    from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

    if compute_backend not in {"cpu", "local_gpu"}:
        raise ValueError("Local image pipeline requires the CPU or System GPU backend.")
    if compute_backend == "local_gpu" and not torch.cuda.is_available():
        raise ValueError("System GPU was selected but CUDA is unavailable to this Python environment.")
    with model_lock:
        use_cuda = compute_backend == "local_gpu"
        device = "cuda" if use_cuda else "cpu"
        if pipe is None or pipeline_mode != mode or pipeline_device != device:
            pipe = None
            pipeline_mode = None
            pipeline_device = None
            active_lora = None
            video_pipe = None
            active_video_lora = None
            gc.collect()
            if use_cuda:
                torch.cuda.empty_cache()
            local_model = MODEL_DIR if (MODEL_DIR / "model_index.json").is_file() else MODEL_ID
            pipeline_class = StableDiffusionImg2ImgPipeline if mode == "img2img" else StableDiffusionPipeline
            kwargs = {
                "torch_dtype": torch.float16 if use_cuda else torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            if local_model == MODEL_DIR:
                kwargs["local_files_only"] = True
            try:
                loaded = pipeline_class.from_pretrained(local_model, **kwargs)
                loaded = loaded.to("cuda" if use_cuda else "cpu")
                if not use_cuda:
                    loaded.enable_attention_slicing("max")
                try:
                    loaded.enable_vae_slicing()
                except Exception:
                    pass
                pipe = loaded
                pipeline_mode = mode
                pipeline_device = device
                model_error = None
            except Exception as exc:
                model_error = str(exc)
                raise
        if active_lora != profile_id:
            if active_lora is not None:
                pipe.unload_lora_weights()
                active_lora = None
            if profile_id:
                path = model_path(profile_id)
                if not profile_ready_for_backend(profile_id, "local") or not path.is_dir():
                    raise ValueError("This face profile has no locally trained LoRA. Train it locally or turn on RunPod.")
                pipe.load_lora_weights(str(path))
                active_lora = profile_id
        return pipe


@app.post("/generate")
def generate():
    try:
        prompt = _clean_prompt(request.form.get("prompt"), "Prompt", 1200)
        negative_prompt = _clean_prompt(request.form.get("negative_prompt"), "Negative prompt", 600)
        profile_id = validate_profile_id(request.form.get("face_profile_id")) if request.form.get("face_profile_id") else None
        strength = float(request.form.get("strength", "0.45"))
        steps = int(request.form.get("steps", "10"))
        title = _clean_prompt(request.form.get("title"), "Title", 160)
        backend = _resolve_compute_backend(request.form)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc) or "Invalid generation settings."), 400
    if not prompt:
        return jsonify(error="Enter a prompt describing the image you want."), 400
    if not 0.2 <= strength <= 0.95 or not 5 <= steps <= 25:
        return jsonify(error="Strength must be 0.20–0.95 and steps must be 5–25."), 400
    profile = get_profile(profile_id) if profile_id else None
    if profile_id and not profile:
        return jsonify(error="Face profile not found."), 404

    image_file = request.files.get("image")
    try:
        init_image = None
        if image_file and image_file.filename:
            with Image.open(image_file.stream) as source:
                init_image = ImageOps.exif_transpose(source).convert("RGB")
                init_image.load()
            init_image = ImageOps.fit(init_image, (512, 512), method=Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error="Use a valid PNG, JPG or WEBP reference image."), 400

    if not generation_lock.acquire(blocking=False):
        return jsonify(error="Another generation is running. Please wait."), 429
    try:
        if backend == "runpod":
            if profile_id and not profile.get("runpod_model_ready"):
                return jsonify(error="This face profile has no RunPod-trained model. Select Train and choose RunPod GPU first."), 400
            data = {
                "prompt": prompt, "negative_prompt": negative_prompt,
                "steps": str(steps), "strength": str(strength),
            }
            if profile_id:
                data["face_profile_id"] = profile_id
            files = None
            if init_image is not None:
                buffer = io.BytesIO()
                init_image.save(buffer, format="PNG")
                buffer.seek(0)
                files = {"image": ("reference.png", buffer, "image/png")}
            try:
                response = runpod_request(
                    "POST", "/api/remote/generate", data=data, files=files, timeout=(20, 600)
                )
            except RuntimeError as exc:
                return jsonify(error=str(exc)), 502
            filename = f"{uuid.uuid4().hex}.png"
            (OUTPUT_DIR / filename).write_bytes(response.content)
            try:
                asset = _persist_generated_asset(
                    "image", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                    profile=profile, backend="runpod", generation_mode="img2img" if init_image is not None else "txt2img",
                )
            except RuntimeError as exc:
                return jsonify(error=str(exc)), 503
            return jsonify(
                status="completed", image_url=f"/assets/{asset['id']}/content", filename=filename,
                asset_id=asset["id"], title=asset["title"],
                mode="img2img" if init_image is not None else "txt2img",
                face_profile=profile["name"] if profile else None,
                backend="runpod", message="Generated on the selected RunPod GPU and saved to the database library.",
            )

        if profile_id and not profile_ready_for_backend(profile_id, "local"):
            return jsonify(error="This face profile has no locally trained LoRA. Train it using the System GPU or select RunPod GPU."), 400
        try:
            mode = "img2img" if init_image is not None else "txt2img"
            pipeline = _load_cpu_or_gpu_pipeline(mode, profile_id, backend)
            generator_device = "cuda" if backend == "local_gpu" else "cpu"
            kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or None,
                "num_inference_steps": steps,
                "guidance_scale": 7.0,
                "generator": torch.Generator(device=generator_device).manual_seed(int(time.time_ns() % (2**31 - 1))),
            }
            if init_image is not None:
                kwargs.update(image=init_image, strength=strength)
            result = pipeline(**kwargs).images[0]
        except Exception as exc:
            message = str(exc)
            if "out of memory" in message.lower() or "memory" in message.lower():
                message = "The selected device ran out of memory. Try fewer steps or switch to another available compute backend."
            return jsonify(error=f"Generation failed: {message}"), 500
        filename = _save_generated_png(result)
        try:
            asset = _persist_generated_asset(
                "image", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                profile=profile, backend=backend, generation_mode="img2img" if init_image is not None else "txt2img",
            )
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503
        if backend == "local_gpu":
            message = "Generated locally on the System GPU and saved to the database library."
        else:
            message = "Generated locally on this computer's CPU and saved to the database library."
        return jsonify(
            status="completed", image_url=f"/assets/{asset['id']}/content", filename=filename,
            asset_id=asset["id"], title=asset["title"],
            mode="img2img" if init_image is not None else "txt2img",
            face_profile=profile["name"] if profile else None,
            backend=backend, message=message,
        )
    finally:
        generation_lock.release()


@app.post("/generate-video")
def generate_video():
    prompt = _clean_prompt(request.form.get("prompt"), "Prompt", 1000)
    negative_prompt = _clean_prompt(request.form.get("negative_prompt"), "Negative prompt", 600)
    try:
        profile_id = validate_profile_id(request.form.get("face_profile_id")) if request.form.get("face_profile_id") else None
        steps = int(request.form.get("steps", "12"))
        frames = int(request.form.get("frames", "16"))
        title = _clean_prompt(request.form.get("title"), "Title", 160)
        backend = _resolve_compute_backend(request.form)
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc) or "Invalid video settings."), 400
    if not prompt:
        return jsonify(error="Enter a video prompt."), 400
    if not 5 <= steps <= 20 or frames not in (8, 16, 24):
        return jsonify(error="Choose 5–20 steps and 8, 16, or 24 frames."), 400
    profile = get_profile(profile_id) if profile_id else None
    if profile_id and not profile:
        return jsonify(error="Face profile not found."), 404
    if backend == "cpu":
        return jsonify(error="Video generation requires a GPU. Turn on GPU acceleration and select an available System GPU or RunPod GPU."), 400

    if not generation_lock.acquire(blocking=False):
        return jsonify(error="Another generation is running. Please wait."), 429
    try:
        if backend == "runpod":
            if profile_id and not profile.get("runpod_model_ready"):
                return jsonify(error="This face profile has no RunPod-trained model. Train it using RunPod GPU first."), 400
            data = {"prompt": prompt, "negative_prompt": negative_prompt, "steps": str(steps), "frames": str(frames)}
            if profile_id:
                data["face_profile_id"] = profile_id
            try:
                response = runpod_request(
                    "POST", "/api/remote/generate-video", data=data, timeout=(20, 1200)
                )
            except RuntimeError as exc:
                return jsonify(error=str(exc)), 502
            filename = f"{uuid.uuid4().hex}.mp4"
            (VIDEO_OUTPUT_DIR / filename).write_bytes(response.content)
            try:
                asset = _persist_generated_asset(
                    "video", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                    profile=profile, backend="runpod", generation_mode="animatediff",
                )
            except RuntimeError as exc:
                return jsonify(error=str(exc)), 503
            return jsonify(
                status="completed", video_url=f"/assets/{asset['id']}/content", filename=filename,
                asset_id=asset["id"], title=asset["title"],
                face_profile=profile["name"] if profile else None, backend="runpod",
                message="Generated on the selected RunPod GPU and saved to the database library.",
            )

        if backend != "local_gpu" or not torch.cuda.is_available():
            return jsonify(error="System GPU is not available to this Python environment. Select RunPod GPU."), 400
        if profile_id and not profile_ready_for_backend(profile_id, "local"):
            return jsonify(error="This face profile has no locally trained LoRA. Train it using the System GPU or select RunPod GPU."), 400
        if not (MOTION_DIR / "config.json").is_file():
            return jsonify(error="AnimateDiff motion adapter is missing. Run download_video_model.bat first."), 400
        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter
            from diffusers.utils import export_to_video
            global video_pipe, active_video_lora, video_model_error
            with model_lock:
                if video_pipe is None:
                    adapter = MotionAdapter.from_pretrained(str(MOTION_DIR), local_files_only=True, torch_dtype=torch.float16)
                    video_pipe = AnimateDiffPipeline.from_pretrained(
                        str(MODEL_DIR), motion_adapter=adapter, local_files_only=True,
                        torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False,
                    )
                    video_pipe.enable_vae_slicing()
                    video_pipe.to("cuda")
                    active_video_lora = None
                if active_video_lora != profile_id:
                    if active_video_lora is not None:
                        video_pipe.unload_lora_weights()
                        active_video_lora = None
                    if profile_id:
                        video_pipe.load_lora_weights(str(model_path(profile_id)))
                        active_video_lora = profile_id
            generator = torch.Generator(device="cuda").manual_seed(int(time.time_ns() % (2**31 - 1)))
            result = video_pipe(
                prompt=prompt, negative_prompt=negative_prompt or None,
                num_frames=frames, num_inference_steps=steps,
                guidance_scale=7.0, generator=generator,
            )
            filename = f"{uuid.uuid4().hex}.mp4"
            export_to_video(result.frames[0], str(VIDEO_OUTPUT_DIR / filename), fps=8)
        except Exception as exc:
            video_model_error = str(exc)
            message = str(exc)
            if "out of memory" in message.lower():
                message = "The GPU ran out of memory. Try fewer frames or use the RunPod GPU."
            return jsonify(error=f"Video generation failed: {message}"), 500
        try:
            asset = _persist_generated_asset(
                "video", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                profile=profile, backend="local_gpu", generation_mode="animatediff",
            )
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503
        return jsonify(
            status="completed", video_url=f"/assets/{asset['id']}/content", filename=filename,
            asset_id=asset["id"], title=asset["title"],
            face_profile=profile["name"] if profile else None, backend="local_gpu",
            message="Generated locally on the System GPU and saved to the database library.",
        )
    finally:
        generation_lock.release()


@app.get("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.get("/videos/<path:filename>")
def videos(filename):
    return send_from_directory(VIDEO_OUTPUT_DIR, filename, as_attachment=False, mimetype="video/mp4")


if __name__ == "__main__":
    print("\nLocal Private Image AI")
    print("Open http://127.0.0.1:5000 in your browser.")
    print("Only this computer can access the app. Do not change host to 0.0.0.0.")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
