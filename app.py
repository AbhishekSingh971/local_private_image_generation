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
from image_generation import build_generation_prompts, resolve_image_dimensions, validate_image_preset
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
    count_assets,
    create_asset,
    database_status,
    database_url_configured,
    ensure_schema,
    create_face_dataset_group,
    delete_face_dataset_group,
    delete_face_photo,
    get_face_dataset_group,
    get_face_photo_bytes,
    list_face_dataset_group_photo_records,
    get_face_profile_record,
    insert_face_profile_record,
    list_face_dataset_groups,
    update_face_dataset_group,
    update_face_photo,
    get_face_model_bundle,
    list_asset_filenames,
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
MODEL_ID = os.environ.get("SD_MODEL_ID", "").strip() or "stable-diffusion-v1-5/stable-diffusion-v1-5"
MODEL_PATH_OVERRIDE = os.environ.get("SD_MODEL_PATH", "").strip()

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
LEGACY_MEDIA_MIGRATION_LOCK = threading.Lock()
LEGACY_MEDIA_MIGRATED = False
GENERATION_PROGRESS = {}
GENERATION_PROGRESS_LOCK = threading.RLock()


def _set_generation_progress(task_id, *, asset_type, backend, status, progress, current_step=0, total_steps=0, stage="", message=""):
    if not re.fullmatch(r"[a-f0-9]{32}", str(task_id or "")):
        return
    now = time.time()
    record = {
        "task_id": task_id, "asset_type": asset_type, "backend": backend,
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


def _get_generation_progress(task_id):
    with GENERATION_PROGRESS_LOCK:
        progress = GENERATION_PROGRESS.get(task_id)
        return dict(progress) if progress else None


@app.get("/api/generation-progress/<task_id>")
def api_generation_progress(task_id):
    if not re.fullmatch(r"[a-f0-9]{32}", task_id or ""):
        return jsonify(error="Invalid generation task ID."), 400
    progress = _get_generation_progress(task_id)
    if not progress:
        return jsonify(task_id=task_id, status="starting", progress=0, current_step=0,
                       total_steps=0, stage="starting", message="Waiting for the generation worker…")
    if progress.get("backend") == "runpod" and progress.get("status") not in {"completed", "failed"}:
        try:
            response = runpod_request(
                "GET", f"/api/remote/generation-progress/{task_id}", timeout=(3, 5)
            )
            remote_progress = response.json()
            if remote_progress.get("task_id") == task_id:
                # Remote completion means the GPU finished; the local app still needs
                # to persist the returned bytes in PostgreSQL before the task is complete.
                remote_status = remote_progress.get("status", "running")
                remote_percent = remote_progress.get("progress", 0)
                remote_stage = remote_progress.get("stage", "")
                remote_message = remote_progress.get("message", "")
                if remote_status == "completed":
                    # The worker finished rendering, but the HTTP response body may
                    # still be transferring to this app instance. Do not move backwards
                    # if the local app has already advanced to database persistence.
                    remote_status = "running"
                    if progress.get("stage") == "saving":
                        remote_percent = progress.get("progress", 96)
                        remote_stage = "saving"
                        remote_message = progress.get("message", "Saving media to the library…")
                    else:
                        remote_percent = max(93, int(progress.get("progress", 0)))
                        remote_stage = "transferring"
                        remote_message = "GPU work finished; receiving the rendered file from RunPod…"
                _set_generation_progress(
                    task_id, asset_type=progress["asset_type"], backend="runpod",
                    status=remote_status, progress=remote_percent,
                    current_step=remote_progress.get("current_step", 0),
                    total_steps=remote_progress.get("total_steps", 0),
                    stage=remote_stage, message=remote_message,
                )
                progress = _get_generation_progress(task_id) or progress
        except Exception:
            # Generation remains active even if a status poll fails momentarily.
            pass
    return jsonify(progress)


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


def _migrate_legacy_output_files():
    """Import output files from earlier versions into the persistent asset library once."""
    global LEGACY_MEDIA_MIGRATED
    if LEGACY_MEDIA_MIGRATED:
        return
    with LEGACY_MEDIA_MIGRATION_LOCK:
        if LEGACY_MEDIA_MIGRATED:
            return
        try:
            for asset_type, folder, extensions, mime_type in (
                ("image", OUTPUT_DIR, {".png", ".jpg", ".jpeg", ".webp"}, None),
                ("video", VIDEO_OUTPUT_DIR, {".mp4", ".webm", ".mov"}, None),
            ):
                known = list_asset_filenames(asset_type)
                for path in sorted(folder.iterdir()):
                    if not path.is_file() or path.suffix.lower() not in extensions or path.name in known:
                        continue
                    if path.stat().st_size <= 0 or path.stat().st_size > 512 * 1024 * 1024:
                        app.logger.warning("Skipped legacy media file outside the supported size range: %s", path.name)
                        continue
                    detected_mime = (
                        "image/png" if path.suffix.lower() == ".png" else
                        "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else
                        "image/webp" if path.suffix.lower() == ".webp" else
                        "video/mp4" if path.suffix.lower() == ".mp4" else
                        "video/webm" if path.suffix.lower() == ".webm" else
                        "video/quicktime"
                    )
                    try:
                        create_asset(
                            asset_type=asset_type,
                            title=f"Imported {asset_type} · {path.stem[:80]}",
                            filename=path.name, mime_type=detected_mime,
                            media_data=path.read_bytes(),
                            prompt="", negative_prompt="", profile_id=None, profile_name=None,
                            backend="legacy_import", generation_mode="legacy",
                        )
                        known.add(path.name)
                    except Exception:
                        app.logger.exception("Could not import legacy media file %s", path.name)
                        return
            LEGACY_MEDIA_MIGRATED = True
        except Exception:
            app.logger.exception("Could not inspect legacy output folders for database migration")


def _safe_list_assets(asset_type=None, limit=200):
    try:
        _migrate_legacy_output_files()
        return list_assets(asset_type, limit)
    except Exception:
        app.logger.exception("Media library is unavailable")
        return []


def _safe_asset_count(asset_type):
    try:
        _migrate_legacy_output_files()
        return count_assets(asset_type)
    except Exception:
        app.logger.exception("Media count is unavailable")
        return 0


def _safe_profile_count():
    try:
        return len(list_profiles())
    except Exception:
        app.logger.exception("Face profile store is unavailable")
        return 0


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
    _migrate_legacy_output_files()
    return render_template(
        "dashboard.html", page_key="dashboard", active_page="dashboard",
        image_count=_safe_asset_count("image"), video_count=_safe_asset_count("video"),
        profile_count=_safe_profile_count(), recent_assets=_safe_list_assets(limit=8),
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


@app.get("/training-data")
def training_data_page():
    return render_template("training_data.html", page_key="training-data", active_page="training-data")


@app.get("/api/assets")
def api_list_assets():
    asset_type = (request.args.get("type") or "").strip().lower()
    if asset_type and asset_type not in {"image", "video"}:
        return jsonify(error="Asset type must be image or video."), 400
    try:
        limit = int(request.args.get("limit", "30"))
        offset = int(request.args.get("offset", "0"))
        sort = (request.args.get("sort") or "newest").strip().lower()
        search = (request.args.get("q") or "").strip()[:160]
        if not 1 <= limit <= 500:
            return jsonify(error="Limit must be between 1 and 500."), 400
        if offset < 0:
            return jsonify(error="Offset must be zero or greater."), 400
        if sort not in {"newest", "oldest", "title"}:
            return jsonify(error="Sort must be newest, oldest or title."), 400
        _migrate_legacy_output_files()
        assets = list_assets(asset_type or None, limit, offset, search, sort)
        total = count_assets(asset_type or None, search)
        for asset in assets:
            asset["content_url"] = url_for("asset_content", asset_id=asset["id"])
        return jsonify(
            assets=assets, total=total, limit=limit, offset=offset,
            has_more=offset + len(assets) < total,
        )
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
        existing = get_asset(asset_id)
    except Exception:
        app.logger.exception("Could not read media before deletion")
        return jsonify(error="Could not read the media record from the database."), 503
    if not existing:
        return jsonify(error="Saved media not found."), 404
    folder = OUTPUT_DIR if existing["asset_type"] == "image" else VIDEO_OUTPUT_DIR
    local_copy = folder / Path(existing["filename"]).name
    try:
        if local_copy.is_file():
            local_copy.unlink()
    except OSError:
        app.logger.exception("Could not remove local copy of media before deletion")
        return jsonify(error="Could not remove the local media copy. The database record was not deleted."), 500
    try:
        asset = delete_asset(asset_id)
    except Exception:
        app.logger.exception("Could not delete media from database")
        return jsonify(error="The local copy was removed, but database deletion failed. The saved media remains in the database."), 503
    if not asset:
        return jsonify(error="Saved media not found."), 404
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


@app.post("/api/face-profiles/<profile_id>/dataset-groups")
def api_create_face_dataset_group(profile_id):
    group_id = None
    try:
        profile_id = validate_profile_id(profile_id)
        profile = get_profile(profile_id)
        if not profile:
            return jsonify(error="Face profile not found."), 404
        title = _clean_prompt(request.form.get("title"), "Dataset group title", 120)
        uploads = request.files.getlist("source_image")
        if not uploads or not uploads[0] or not uploads[0].filename:
            return jsonify(error="Choose one source image from your computer."), 400
        images = _parse_uploads(uploads[:1])
        if not images:
            return jsonify(error="The source image could not be read."), 400
        # In local development mode profiles are still in the legacy JSON registry.
        # Mirror profile metadata into SQLite so grouped photos can be browsed too.
        ensure_schema()
        if not get_face_profile_record(profile_id):
            insert_face_profile_record(profile)
        group = create_face_dataset_group(profile_id, title, expected_count=5)
        group_id = group["id"]
        image = images[0]
        if image.width < 128 or image.height < 128:
            raise ValueError("The source image must be at least 128 × 128 pixels.")
        save_profile_image(profile_id, image, group_id=group_id, angle_label="Original source")
        return jsonify(status="created", group=get_face_dataset_group(group_id)), 201
    except ValueError as exc:
        if group_id:
            update_face_dataset_group(group_id, status="failed", error=str(exc))
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Could not create grouped face-training dataset")
        if group_id:
            try:
                update_face_dataset_group(group_id, status="failed", error="Could not save the source image.")
            except Exception:
                pass
        return jsonify(error="Could not create the training-data group. Check database connectivity and profile storage."), 503


@app.get("/api/face-dataset-groups")
def api_list_face_dataset_groups():
    try:
        profile_id = request.args.get("profile_id", "").strip() or None
        if profile_id:
            profile_id = validate_profile_id(profile_id)
            if not get_profile(profile_id):
                return jsonify(error="Face profile not found."), 404
        groups = list_face_dataset_groups(profile_id=profile_id)
        for group in groups:
            for photo in group["photos"]:
                photo["content_url"] = url_for("face_photo_content", photo_id=photo["id"])
        return jsonify(groups=groups, count=len(groups))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Could not list face-training dataset groups")
        return jsonify(error="Could not read dataset groups. Check the database connection."), 503


def _stage_profile_dataset_files(profile_id: str, filenames: list[str]) -> list[tuple[Path, Path]]:
    """Move dataset photos out of the training scan path until a DB delete commits."""
    folder = dataset_dir(validate_profile_id(profile_id)).resolve()
    staged: list[tuple[Path, Path]] = []
    try:
        for filename in filenames:
            name = Path(str(filename or "")).name
            if not name or name in {".", ".."}:
                continue
            source = folder / name
            if source.parent.resolve() != folder or not source.is_file():
                continue
            temporary = folder / ("." + name + "." + uuid.uuid4().hex + ".delete")
            source.replace(temporary)
            staged.append((source, temporary))
        return staged
    except Exception:
        for source, temporary in reversed(staged):
            try:
                if temporary.exists() and not source.exists():
                    temporary.replace(source)
            except OSError:
                app.logger.exception("Could not restore a staged training photo after deletion preparation failed")
        raise


def _restore_staged_profile_files(staged: list[tuple[Path, Path]]) -> None:
    for source, temporary in reversed(staged):
        try:
            if temporary.exists() and not source.exists():
                temporary.replace(source)
        except OSError:
            app.logger.exception("Could not restore a staged training photo after a database failure")


def _discard_staged_profile_files(staged: list[tuple[Path, Path]]) -> None:
    for _source, temporary in staged:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # The temporary extension is not recognized as a training image.
            app.logger.exception("Could not remove a staged training photo after successful database deletion")


@app.patch("/api/face-dataset-groups/<group_id>")
def api_update_face_dataset_group(group_id):
    if not re.fullmatch(r"[a-f0-9]{32}", group_id or ""):
        return jsonify(error="Invalid dataset group ID."), 400
    payload = request.get_json(silent=True) or {}
    try:
        existing_group = get_face_dataset_group(group_id)
        if not existing_group:
            return jsonify(error="Dataset group not found."), 404
        if "title" in payload and existing_group.get("status") == "generating":
            return jsonify(error="Wait for generation to finish before renaming this group."), 409
        group = update_face_dataset_group(
            group_id,
            title=payload.get("title") if "title" in payload else None,
            status=payload.get("status"),
            generated_count=payload.get("generated_count"),
            expected_count=payload.get("expected_count"),
            error=payload.get("error") if "error" in payload else None,
        )
        if not group:
            return jsonify(error="Dataset group not found."), 404
        return jsonify(group=group)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    except Exception:
        app.logger.exception("Could not update dataset group status")
        return jsonify(error="Could not update dataset group status."), 503


@app.delete("/api/face-dataset-groups/<group_id>")
def api_delete_face_dataset_group(group_id):
    if not re.fullmatch(r"[a-f0-9]{32}", group_id or ""):
        return jsonify(error="Invalid dataset group ID."), 400
    existing_group = get_face_dataset_group(group_id)
    if not existing_group:
        return jsonify(error="Dataset group not found."), 404
    if existing_group.get("status") == "generating":
        return jsonify(error="This dataset group is still generating. Wait for it to finish before deleting it."), 409
    try:
        photo_records = list_face_dataset_group_photo_records(group_id)
        staged = _stage_profile_dataset_files(
            existing_group["profile_id"],
            [photo["filename"] for photo in photo_records],
        )
    except Exception:
        app.logger.exception("Could not stage local photo files for dataset-group deletion")
        return jsonify(error="Could not prepare the group's local training photos for deletion."), 500
    try:
        deleted = delete_face_dataset_group(group_id)
    except Exception:
        _restore_staged_profile_files(staged)
        app.logger.exception("Could not delete dataset group from the database")
        return jsonify(error="Could not delete the dataset group from the database."), 503
    if not deleted:
        _restore_staged_profile_files(staged)
        return jsonify(error="Dataset group not found."), 404
    _discard_staged_profile_files(staged)
    return jsonify(status="deleted", id=group_id, deleted_photos=len(deleted.get("photos") or []))


@app.route("/api/face-photos/<photo_id>", methods=["PATCH", "PUT", "DELETE"])
def api_face_photo_detail(photo_id):
    if not re.fullmatch(r"[a-f0-9]{32}", photo_id or ""):
        return jsonify(error="Invalid face photo ID."), 400
    try:
        stored = get_face_photo_bytes(photo_id)
    except Exception:
        app.logger.exception("Could not read grouped face photo before update")
        return jsonify(error="Could not read the photo record from storage."), 503
    if not stored:
        return jsonify(error="Training photo not found."), 404
    metadata, _old_bytes = stored
    group_id = metadata.get("group_id")
    if not group_id:
        return jsonify(error="Only photos inside a named training group can be managed here."), 400
    group = get_face_dataset_group(group_id)
    if not group:
        return jsonify(error="The photo's dataset group no longer exists."), 404
    if group.get("status") == "generating":
        return jsonify(error="This group is still generating. Wait before editing its photos."), 409

    if request.method == "PATCH":
        payload = request.get_json(silent=True) or {}
        if "angle_label" not in payload:
            return jsonify(error="Supply an angle_label to update the photo label."), 400
        try:
            photo = update_face_photo(photo_id, angle_label=payload.get("angle_label"))
            if not photo:
                return jsonify(error="Training photo not found."), 404
            return jsonify(status="updated", photo=photo)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception("Could not update training photo label")
            return jsonify(error="Could not update the training photo label."), 503

    if request.method == "PUT":
        upload = request.files.get("photo")
        try:
            images = _parse_uploads([upload] if upload and upload.filename else [], max_files=1)
            if not images:
                return jsonify(error="Choose a replacement photo file."), 400
            buffer = io.BytesIO()
            images[0].save(buffer, format="JPEG", quality=95, optimize=True)
            image_data = buffer.getvalue()
            destination = dataset_dir(metadata["profile_id"]) / Path(metadata["filename"]).name
            if destination.parent.resolve() != dataset_dir(metadata["profile_id"]).resolve():
                return jsonify(error="Invalid training photo path."), 400
            temporary = destination.with_name("." + destination.name + "." + uuid.uuid4().hex + ".tmp")
            temporary.write_bytes(image_data)
            try:
                photo = update_face_photo(photo_id, image_data=image_data, mime_type="image/jpeg")
                if not photo:
                    raise ValueError("Training photo not found.")
                temporary.replace(destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            return jsonify(status="replaced", photo=photo)
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        except Exception:
            app.logger.exception("Could not replace a training photo")
            return jsonify(error="Could not replace the training photo in storage."), 503

    try:
        staged = _stage_profile_dataset_files(metadata["profile_id"], [metadata["filename"]])
    except Exception:
        app.logger.exception("Could not stage training photo for deletion")
        return jsonify(error="Could not prepare the local training photo for deletion."), 500
    try:
        deleted = delete_face_photo(photo_id)
    except Exception:
        _restore_staged_profile_files(staged)
        app.logger.exception("Could not delete training photo from the database")
        return jsonify(error="Could not delete the training photo from storage."), 503
    if not deleted:
        _restore_staged_profile_files(staged)
        return jsonify(error="Training photo not found."), 404
    _discard_staged_profile_files(staged)
    angle_label = (metadata.get("angle_label") or "").strip()
    generated_angles = {"Front view", "Three-quarter left", "Three-quarter right", "Left profile", "Right profile"}
    if angle_label in generated_angles:
        new_count = max(0, int(group.get("generated_count") or 0) - 1)
        next_status = "partial" if new_count < int(group.get("expected_count") or 5) else group.get("status")
        try:
            update_face_dataset_group(
                group_id, generated_count=new_count, status=next_status, error=""
            )
        except Exception:
            app.logger.exception("Could not recalculate grouped photo count after deletion")
    return jsonify(status="deleted", id=photo_id, group_id=group_id)


@app.get("/api/face-photos/<photo_id>/content")
def face_photo_content(photo_id):
    if not re.fullmatch(r"[a-f0-9]{32}", photo_id or ""):
        return jsonify(error="Invalid face photo ID."), 400
    try:
        stored = get_face_photo_bytes(photo_id)
    except Exception:
        app.logger.exception("Could not read grouped face photo")
        return jsonify(error="Could not read the grouped photo from storage."), 503
    if not stored:
        return jsonify(error="Grouped face photo not found."), 404
    metadata, image_data = stored
    return send_file(io.BytesIO(image_data), mimetype=metadata["mime_type"], max_age=0)


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
    group_id = str(payload.get("group_id") or "").strip() or None
    angle_labels = payload.get("angle_labels") or []
    if not isinstance(asset_ids, list) or not asset_ids:
        return jsonify(error="Select one or more generated images."), 400
    if group_id:
        if not re.fullmatch(r"[a-f0-9]{32}", group_id):
            return jsonify(error="Invalid dataset group ID."), 400
        group = get_face_dataset_group(group_id)
        if not group or group["profile_id"] != profile_id:
            return jsonify(error="Dataset group not found for this face profile."), 404
        if not isinstance(angle_labels, list) or len(angle_labels) != len(asset_ids):
            return jsonify(error="Supply one angle label for each generated image."), 400
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
            label = str(angle_labels[added] or "")[:80] if group_id else None
            save_profile_image(
                profile_id, image, group_id=group_id, angle_label=label
            )
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
        description = request.form.get("description") or ""
        uploads = request.files.getlist("photos")
        images = _parse_uploads(uploads) if any(item and item.filename for item in uploads) else []
        profile = create_profile(name, token=token, description=description)
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
        group_id = (request.form.get("group_id") or "").strip() or None
        if group_id:
            group = get_face_dataset_group(group_id)
            if not group or group["profile_id"] != profile_id:
                return jsonify(error="Dataset group not found for this profile."), 404
            if group.get("status") == "generating":
                return jsonify(error="Wait for angle generation to finish before adding more photos to this group."), 409
        uploads = [item for item in request.files.getlist("photos") if item and item.filename]
        images = _parse_uploads(uploads)
        if not images:
            return jsonify(error="Choose one or more photos to upload."), 400
        labels = request.form.getlist("labels")
        for index, image in enumerate(images):
            label = None
            if group_id:
                label = (labels[index].strip() if index < len(labels) else "") or f"Additional photo {index + 1}"
            save_profile_image(profile_id, image, group_id=group_id, angle_label=label)
        if group_id:
            update_face_dataset_group(group_id, status="partial", error="")
        return jsonify(status="saved", profile=get_profile(profile_id), added=len(images), group_id=group_id)
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
                    "description": profile.get("description", ""),
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
        if database_url_configured() and profile.get("runpod_training_state") != "completed":
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
            # Explicit model configuration takes priority over the bundled SD 1.5 folder.
            # This lets the user select a compatible, properly licensed photorealistic checkpoint.
            if MODEL_PATH_OVERRIDE:
                override_path = Path(MODEL_PATH_OVERRIDE).expanduser()
                if not override_path.is_dir() or not (override_path / "model_index.json").is_file():
                    raise ValueError("SD_MODEL_PATH must point to an existing Diffusers model directory containing model_index.json.")
                local_model = override_path
            elif os.environ.get("SD_MODEL_ID", "").strip():
                local_model = MODEL_ID
            else:
                local_model = MODEL_DIR if (MODEL_DIR / "model_index.json").is_file() else MODEL_ID
            pipeline_class = StableDiffusionImg2ImgPipeline if mode == "img2img" else StableDiffusionPipeline
            kwargs = {
                "torch_dtype": torch.float16 if use_cuda else torch.float32,
                "safety_checker": None,
                "requires_safety_checker": False,
            }
            if isinstance(local_model, Path):
                if not (local_model / "model_index.json").is_file():
                    raise ValueError(f"SD_MODEL_PATH is not a Diffusers model folder: {local_model}")
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
        steps = int(request.form.get("steps", "25"))
        title = _clean_prompt(request.form.get("title"), "Title", 160)
        aspect_ratio = (request.form.get("aspect_ratio") or "portrait").strip().lower()
        image_preset = validate_image_preset(request.form.get("image_preset") or "full_body_photo")
        width, height = resolve_image_dimensions(aspect_ratio)
        backend = _resolve_compute_backend(request.form)
        task_id = (request.form.get("task_id") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid generation task ID. Refresh the page and retry.")
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
            init_image = ImageOps.fit(init_image, (width, height), method=Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return jsonify(error="Use a valid PNG, JPG or WEBP reference image."), 400

    _set_generation_progress(
        task_id, asset_type="image", backend=backend, status="running", progress=1,
        current_step=0, total_steps=steps, stage="starting", message="Preparing your image request…",
    )
    if not generation_lock.acquire(blocking=False):
        _set_generation_progress(
            task_id, asset_type="image", backend=backend, status="failed", progress=0,
            total_steps=steps, stage="busy", message="Another generation is already running.",
        )
        return jsonify(error="Another generation is running. Please wait."), 429
    try:
        if backend == "runpod":
            if profile_id and not profile.get("runpod_model_ready"):
                return jsonify(error="This face profile has no RunPod-trained model. Select Train and choose RunPod GPU first."), 400
            _set_generation_progress(
                task_id, asset_type="image", backend="runpod", status="running", progress=2,
                total_steps=steps, stage="connecting", message="Connecting to RunPod GPU worker…",
            )
            data = {
                "prompt": prompt, "negative_prompt": negative_prompt,
                "steps": str(steps), "strength": str(strength), "task_id": task_id,
                "aspect_ratio": aspect_ratio, "image_preset": image_preset,
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
                _set_generation_progress(
                    task_id, asset_type="image", backend="runpod", status="failed", progress=0,
                    total_steps=steps, stage="failed", message=str(exc),
                )
                return jsonify(error=str(exc)), 502
            _set_generation_progress(
                task_id, asset_type="image", backend="runpod", status="running", progress=96,
                total_steps=steps, stage="saving", message="GPU generation finished; saving image to the library…",
            )
            filename = f"{uuid.uuid4().hex}.png"
            (OUTPUT_DIR / filename).write_bytes(response.content)
            try:
                asset = _persist_generated_asset(
                    "image", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                    profile=profile, backend="runpod", generation_mode="img2img" if init_image is not None else "txt2img",
                )
            except RuntimeError as exc:
                _set_generation_progress(
                    task_id, asset_type="image", backend="runpod", status="failed", progress=96,
                    total_steps=steps, stage="failed", message=str(exc),
                )
                return jsonify(error=str(exc)), 503
            _set_generation_progress(
                task_id, asset_type="image", backend="runpod", status="completed", progress=100,
                current_step=steps, total_steps=steps, stage="completed", message="Image saved to the library.",
            )
            return jsonify(
                status="completed", task_id=task_id, image_url=f"/assets/{asset['id']}/content", filename=filename,
                asset_id=asset["id"], title=asset["title"],
                mode="img2img" if init_image is not None else "txt2img",
                face_profile=profile["name"] if profile else None,
                backend="runpod", message="Generated on the selected RunPod GPU and saved to the database library.",
            )

        if profile_id and not profile_ready_for_backend(profile_id, "local"):
            return jsonify(error="This face profile has no locally trained LoRA. Train it using the System GPU or select RunPod GPU."), 400
        try:
            mode = "img2img" if init_image is not None else "txt2img"
            _set_generation_progress(
                task_id, asset_type="image", backend=backend, status="running", progress=3,
                total_steps=steps, stage="loading_model", message="Loading the image model and selected face…",
            )
            pipeline = _load_cpu_or_gpu_pipeline(mode, profile_id, backend)
            generator_device = "cuda" if backend == "local_gpu" else "cpu"
            _set_generation_progress(
                task_id, asset_type="image", backend=backend, status="running", progress=5,
                total_steps=steps, stage="denoising", message="Starting image denoising steps…",
            )
            def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
                scheduler_steps = len(getattr(active_pipeline.scheduler, "timesteps", [])) or steps
                current = min(step_index + 1, scheduler_steps)
                percent = min(92, 5 + round(87 * current / max(scheduler_steps, 1)))
                _set_generation_progress(
                    task_id, asset_type="image", backend=backend, status="running", progress=percent,
                    current_step=current, total_steps=scheduler_steps, stage="denoising",
                    message=f"Generating image · step {current} of {scheduler_steps}",
                )
                return callback_kwargs
            effective_prompt, effective_negative_prompt = build_generation_prompts(
                prompt, negative_prompt, image_preset
            )
            kwargs = {
                "prompt": effective_prompt,
                "negative_prompt": effective_negative_prompt or None,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": 7.5,
                "generator": torch.Generator(device=generator_device).manual_seed(int(time.time_ns() % (2**31 - 1))),
            }
            if init_image is not None:
                kwargs.update(image=init_image, strength=strength)
            result = pipeline(**kwargs, callback_on_step_end=on_step_end).images[0]
            _set_generation_progress(
                task_id, asset_type="image", backend=backend, status="running", progress=95,
                current_step=steps, total_steps=steps, stage="finalizing", message="Finalizing the generated image…",
            )
        except Exception as exc:
            _set_generation_progress(
                task_id, asset_type="image", backend=backend, status="failed", progress=0,
                total_steps=steps, stage="failed", message="Image generation failed.",
            )
            message = str(exc)
            if "out of memory" in message.lower() or "memory" in message.lower():
                message = "The selected device ran out of memory. Try fewer steps or switch to another available compute backend."
            return jsonify(error=f"Generation failed: {message}"), 500
        filename = _save_generated_png(result)
        _set_generation_progress(
            task_id, asset_type="image", backend=backend, status="running", progress=97,
            current_step=steps, total_steps=steps, stage="saving", message="Saving image bytes and generation details to the database…",
        )
        try:
            asset = _persist_generated_asset(
                "image", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                profile=profile, backend=backend, generation_mode="img2img" if init_image is not None else "txt2img",
            )
        except RuntimeError as exc:
            _set_generation_progress(
                task_id, asset_type="image", backend=backend, status="failed", progress=96,
                total_steps=steps, stage="failed", message="The image could not be saved to the database.",
            )
            return jsonify(error=str(exc)), 503
        if backend == "local_gpu":
            message = "Generated locally on the System GPU and saved to the database library."
        else:
            message = "Generated locally on this computer's CPU and saved to the database library."
        _set_generation_progress(
            task_id, asset_type="image", backend=backend, status="completed", progress=100,
            current_step=steps, total_steps=steps, stage="completed", message="Image saved to the library.",
        )
        return jsonify(
            status="completed", task_id=task_id, image_url=f"/assets/{asset['id']}/content", filename=filename,
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
        task_id = (request.form.get("task_id") or "").strip().lower()
        if not re.fullmatch(r"[a-f0-9]{32}", task_id):
            raise ValueError("Invalid generation task ID. Refresh the page and retry.")
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

    _set_generation_progress(
        task_id, asset_type="video", backend=backend, status="running", progress=1,
        current_step=0, total_steps=steps, stage="starting", message="Preparing your video request…",
    )
    if not generation_lock.acquire(blocking=False):
        _set_generation_progress(
            task_id, asset_type="video", backend=backend, status="failed", progress=0,
            total_steps=steps, stage="busy", message="Another generation is already running.",
        )
        return jsonify(error="Another generation is running. Please wait."), 429
    try:
        if backend == "runpod":
            if profile_id and not profile.get("runpod_model_ready"):
                return jsonify(error="This face profile has no RunPod-trained model. Train it using RunPod GPU first."), 400
            _set_generation_progress(
                task_id, asset_type="video", backend="runpod", status="running", progress=2,
                total_steps=steps, stage="connecting", message="Connecting to RunPod GPU worker…",
            )
            data = {"prompt": prompt, "negative_prompt": negative_prompt, "steps": str(steps), "frames": str(frames), "task_id": task_id}
            if profile_id:
                data["face_profile_id"] = profile_id
            try:
                response = runpod_request(
                    "POST", "/api/remote/generate-video", data=data, timeout=(20, 1200)
                )
            except RuntimeError as exc:
                _set_generation_progress(
                    task_id, asset_type="video", backend="runpod", status="failed", progress=0,
                    total_steps=steps, stage="failed", message=str(exc),
                )
                return jsonify(error=str(exc)), 502
            _set_generation_progress(
                task_id, asset_type="video", backend="runpod", status="running", progress=96,
                total_steps=steps, stage="saving", message="GPU generation finished; saving video to the library…",
            )
            filename = f"{uuid.uuid4().hex}.mp4"
            (VIDEO_OUTPUT_DIR / filename).write_bytes(response.content)
            try:
                asset = _persist_generated_asset(
                    "video", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                    profile=profile, backend="runpod", generation_mode="animatediff",
                )
            except RuntimeError as exc:
                _set_generation_progress(
                    task_id, asset_type="video", backend="runpod", status="failed", progress=96,
                    total_steps=steps, stage="failed", message=str(exc),
                )
                return jsonify(error=str(exc)), 503
            _set_generation_progress(
                task_id, asset_type="video", backend="runpod", status="completed", progress=100,
                current_step=steps, total_steps=steps, stage="completed", message="Video saved to the library.",
            )
            return jsonify(
                status="completed", task_id=task_id, video_url=f"/assets/{asset['id']}/content", filename=filename,
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
            _set_generation_progress(
                task_id, asset_type="video", backend=backend, status="running", progress=3,
                total_steps=steps, stage="loading_model", message="Loading the video model and selected face…",
            )
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
            _set_generation_progress(
                task_id, asset_type="video", backend=backend, status="running", progress=5,
                total_steps=steps, stage="denoising", message="Starting video denoising steps…",
            )
            def on_step_end(active_pipeline, step_index, _timestep, callback_kwargs):
                scheduler_steps = len(getattr(active_pipeline.scheduler, "timesteps", [])) or steps
                current = min(step_index + 1, scheduler_steps)
                percent = min(92, 5 + round(87 * current / max(scheduler_steps, 1)))
                _set_generation_progress(
                    task_id, asset_type="video", backend=backend, status="running", progress=percent,
                    current_step=current, total_steps=scheduler_steps, stage="denoising",
                    message=f"Generating video · step {current} of {scheduler_steps} · {frames} frames",
                )
                return callback_kwargs
            generator = torch.Generator(device="cuda").manual_seed(int(time.time_ns() % (2**31 - 1)))
            result = video_pipe(
                prompt=prompt, negative_prompt=negative_prompt or None,
                num_frames=frames, num_inference_steps=steps,
                guidance_scale=7.0, generator=generator,
                callback_on_step_end=on_step_end,
            )
            _set_generation_progress(
                task_id, asset_type="video", backend=backend, status="running", progress=95,
                current_step=steps, total_steps=steps, stage="encoding", message="Encoding video frames into MP4…",
            )
            filename = f"{uuid.uuid4().hex}.mp4"
            export_to_video(result.frames[0], str(VIDEO_OUTPUT_DIR / filename), fps=8)
        except Exception as exc:
            _set_generation_progress(
                task_id, asset_type="video", backend=backend, status="failed", progress=0,
                total_steps=steps, stage="failed", message="Video generation failed.",
            )
            video_model_error = str(exc)
            message = str(exc)
            if "out of memory" in message.lower():
                message = "The GPU ran out of memory. Try fewer frames or use the RunPod GPU."
            return jsonify(error=f"Video generation failed: {message}"), 500
        _set_generation_progress(
            task_id, asset_type="video", backend="local_gpu", status="running", progress=97,
            current_step=steps, total_steps=steps, stage="saving", message="Saving video bytes and generation details to the database…",
        )
        try:
            asset = _persist_generated_asset(
                "video", filename, title=title, prompt=prompt, negative_prompt=negative_prompt,
                profile=profile, backend="local_gpu", generation_mode="animatediff",
            )
        except RuntimeError as exc:
            _set_generation_progress(
                task_id, asset_type="video", backend=backend, status="failed", progress=98,
                total_steps=steps, stage="failed", message="The video could not be saved to the database.",
            )
            return jsonify(error=str(exc)), 503
        _set_generation_progress(
            task_id, asset_type="video", backend=backend, status="completed", progress=100,
            current_step=steps, total_steps=steps, stage="completed", message="Video saved to the library.",
        )
        return jsonify(
            status="completed", task_id=task_id, video_url=f"/assets/{asset['id']}/content", filename=filename,
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
