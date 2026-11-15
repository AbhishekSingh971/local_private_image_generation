"""Persistent named face profiles for local and RunPod-backed LoRA workflows.

Profile metadata and face-photo bytes use PostgreSQL whenever DATABASE_URL is configured.
A one-time migration imports the previous JSON registry and any existing profile images.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROFILES_FILE = DATA_DIR / "face_profiles.json"
PROFILES_DIR = DATA_DIR / "face_profiles"
LORA_DIR = ROOT / "models" / "face_loras"
FULL_FINETUNE_DIR = ROOT / "models" / "face_finetunes"
ACTIVE_MODEL_MODE_DIR = ROOT / "models" / "face_model_modes"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROFILE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
GLOBAL_BODY_PROFILE_ID = "ffffffffffffffffffffffffffffffff"
GLOBAL_BODY_PROFILE_NAME = "Shared body model"
GLOBAL_BODY_PROFILE_TOKEN = "globalbody"
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
MAX_DESCRIPTION_WORDS = 200
_LOCK = threading.RLock()


def validate_profile_description(description: str | None) -> str:
    value = (description or "").strip()
    if len(re.findall(r"\S+", value)) > MAX_DESCRIPTION_WORDS:
        raise ValueError(f"Face description must be {MAX_DESCRIPTION_WORDS} words or fewer.")
    return value
_DB_MIGRATED = False

for folder in (DATA_DIR, PROFILES_DIR, LORA_DIR, FULL_FINETUNE_DIR, ACTIVE_MODEL_MODE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


def _database_enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_profile_id(profile_id: str) -> str:
    value = (profile_id or "").strip().lower()
    if not PROFILE_ID_RE.fullmatch(value):
        raise ValueError("Invalid face profile ID.")
    return value


def _load_registry() -> dict:
    if not PROFILES_FILE.exists():
        return {"version": 1, "profiles": []}
    try:
        with PROFILES_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Could not read {PROFILES_FILE}. Back up the file and repair its JSON before continuing."
        ) from exc
    if not isinstance(data, dict) or not isinstance(data.get("profiles"), list):
        raise RuntimeError(f"{PROFILES_FILE} has an unsupported format.")
    return data


def _save_registry(registry: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = PROFILES_FILE.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(registry, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(PROFILES_FILE)


def _find_raw(registry: dict, profile_id: str) -> dict | None:
    for record in registry["profiles"]:
        if record.get("id") == profile_id:
            return record
    return None


def _image_count(profile_id: str) -> int:
    folder = PROFILES_DIR / profile_id / "dataset"
    if not folder.is_dir():
        return 0
    return sum(
        1 for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def _local_lora_ready(profile_id: str) -> bool:
    folder = LORA_DIR / profile_id
    return (
        (folder / "pytorch_lora_weights.safetensors").is_file()
        or (folder / "pytorch_lora_weights.bin").is_file()
    )


def full_finetune_model_path(profile_id: str) -> Path:
    return FULL_FINETUNE_DIR / validate_profile_id(profile_id)


def _local_full_finetune_ready(profile_id: str) -> bool:
    folder = full_finetune_model_path(profile_id)
    return (folder / "config.json").is_file() and (
        (folder / "diffusion_pytorch_model.safetensors").is_file()
        or (folder / "diffusion_pytorch_model.bin").is_file()
    )


def set_profile_model_training_mode(profile_id: str, training_mode: str) -> None:
    """Record which trained artifact is active without changing legacy profile rows."""
    profile_id = validate_profile_id(profile_id)
    mode = str(training_mode or "").strip().lower()
    if mode not in {"lora", "full_finetune"}:
        raise ValueError("Training mode must be lora or full_finetune.")
    mode_dir = ACTIVE_MODEL_MODE_DIR / profile_id
    mode_dir.mkdir(parents=True, exist_ok=True)
    target = mode_dir / "active.json"
    temporary = mode_dir / ("active." + uuid.uuid4().hex + ".tmp")
    payload = {"training_mode": mode, "updated_at": _utc_now()}
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)

    artifact_dir = LORA_DIR / profile_id if mode == "lora" else full_finetune_model_path(profile_id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "training_mode.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def profile_model_training_mode(profile_id: str) -> str:
    """Return the active model type; old profiles continue to default to LoRA."""
    profile_id = validate_profile_id(profile_id)
    marker = ACTIVE_MODEL_MODE_DIR / profile_id / "active.json"
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
        mode = str(data.get("training_mode") or "").strip().lower()
        if mode in {"lora", "full_finetune"}:
            return mode
    except (OSError, ValueError, AttributeError):
        pass

    full_ready = _local_full_finetune_ready(profile_id)
    lora_ready = _local_lora_ready(profile_id)
    if full_ready and not lora_ready:
        return "full_finetune"
    return "lora"


def ensure_full_finetune_model_cached(profile_id: str) -> Path:
    """Lazily restore a large full-tuned SD 1.5 U-Net from the configured Drive remote."""
    profile_id = validate_profile_id(profile_id)
    target = full_finetune_model_path(profile_id)
    if _local_full_finetune_ready(profile_id):
        return target

    remote_root = os.environ.get(
        "FACE_FINETUNE_DRIVE_REMOTE", "gdrive:Local Ai/models/face-finetunes"
    ).strip().rstrip("/")
    rclone = shutil.which("rclone")
    if not rclone:
        raise RuntimeError(
            "The full fine-tuned model is not cached on this GPU worker and rclone is unavailable. "
            "Restore it from the configured Google Drive model archive or retrain it."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / ("." + profile_id + ".restore-" + uuid.uuid4().hex)
    try:
        result = subprocess.run(
            [rclone, "copy", f"{remote_root}/{profile_id}", str(staging),
             "--transfers", "4", "--checkers", "8", "--retries", "2",
             "--low-level-retries", "3", "--stats-one-line"],
            capture_output=True, text=True, timeout=2400, check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "rclone returned an error").strip()[-1200:]
            raise RuntimeError("Could not restore the full fine-tuned model from Google Drive: " + detail)
        if not (
            (staging / "config.json").is_file()
            and ((staging / "diffusion_pytorch_model.safetensors").is_file()
                 or (staging / "diffusion_pytorch_model.bin").is_file())
        ):
            raise RuntimeError(
                "The Google Drive full fine-tune folder is incomplete; the U-Net config or weights are missing."
            )
        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)
        set_profile_model_training_mode(profile_id, "full_finetune")
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _public_record(record: dict) -> dict:
    profile_id = record["id"]
    result = dict(record)
    result["description"] = record.get("description") or ""
    result["image_count"] = max(_image_count(profile_id), int(record.get("image_count", 0) or 0))
    result["local_model_ready"] = _local_lora_ready(profile_id) or _local_full_finetune_ready(profile_id)
    result["runpod_model_ready"] = bool(record.get("runpod_model_ready"))
    result.pop("dataset_dir", None)
    return result


def _restore_database_photos() -> None:
    """Recreate missing local training-cache files from PostgreSQL photo blobs."""
    from studio_db import list_face_photo_names, list_face_photo_records, list_face_profile_records

    for profile in list_face_profile_records():
        folder = dataset_dir(profile["id"])
        existing_files = {item.name for item in folder.iterdir() if item.is_file()}
        missing_names = list_face_photo_names(profile["id"]) - existing_files
        for photo in list_face_photo_records(profile["id"], missing_names):
            filename = Path(photo["filename"]).name
            if not filename or filename in {".", ".."}:
                continue
            destination = folder / filename
            if destination.is_file():
                continue
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            try:
                temporary.write_bytes(photo["image_data"])
                temporary.replace(destination)
                caption = (photo.get("training_caption") or photo.get("angle_label") or "").strip()
                if photo.get("photo_kind") in {"angle", "expression", "body"} and caption:
                    destination.with_name(destination.name + ".caption.txt").write_text(
                        caption[:1200], encoding="utf-8"
                    )
            finally:
                if temporary.exists():
                    temporary.unlink()


def _ensure_database_migrated() -> None:
    """Import legacy JSON metadata/photos once and restore any missing photo cache files."""
    global _DB_MIGRATED
    if not _database_enabled() or _DB_MIGRATED:
        return
    from studio_db import (
        add_face_photo,
        get_face_profile_record,
        insert_face_profile_record,
        list_face_photo_names,
    )
    with _LOCK:
        if _DB_MIGRATED:
            return
        registry = _load_registry()
        # This marker prevents a profile deleted from PostgreSQL from being re-imported
        # from the legacy JSON cache after the next application restart.
        if registry.get("migrated_to_database") != "postgresql":
            for raw in registry["profiles"]:
                profile_id = validate_profile_id(raw.get("id", ""))
                if not get_face_profile_record(profile_id):
                    insert_face_profile_record(raw)
                existing_names = list_face_photo_names(profile_id)
                for image_path in dataset_images(profile_id):
                    if image_path.name not in existing_names:
                        add_face_photo(
                            profile_id, image_path.name, image_path.read_bytes(),
                            "image/jpeg" if image_path.suffix.lower() in {".jpg", ".jpeg"} else "image/" + image_path.suffix.lower().lstrip("."),
                        )
            registry["migrated_to_database"] = "postgresql"
            registry["migrated_at"] = _utc_now()
            _save_registry(registry)
        _restore_database_photos()
        _DB_MIGRATED = True

def list_profiles() -> list[dict]:
    if _database_enabled():
        _ensure_database_migrated()
        from studio_db import list_face_profile_records
        records = list_face_profile_records()
        return [
            _public_record(record) for record in records
            if record.get("id") != GLOBAL_BODY_PROFILE_ID
        ]
    with _LOCK:
        registry = _load_registry()
        return [
            _public_record(record)
            for record in sorted(registry["profiles"], key=lambda item: item.get("created_at", ""))
            if record.get("id") != GLOBAL_BODY_PROFILE_ID
        ]


def get_profile(profile_id: str) -> dict | None:
    profile_id = validate_profile_id(profile_id)
    if _database_enabled():
        _ensure_database_migrated()
        from studio_db import get_face_profile_record
        record = get_face_profile_record(profile_id)
        return _public_record(record) if record else None
    with _LOCK:
        registry = _load_registry()
        record = _find_raw(registry, profile_id)
        return _public_record(record) if record else None


def create_profile(name: str, profile_id: str | None = None, token: str | None = None, description: str = "") -> dict:
    name = (name or "").strip()
    description = validate_profile_description(description)
    if not name:
        raise ValueError("Enter a name for this face profile.")
    if len(name) > 80:
        raise ValueError("Face profile name must be 80 characters or fewer.")
    if any(ord(char) < 32 for char in name):
        raise ValueError("Face profile name contains unsupported control characters.")

    profile_id = validate_profile_id(profile_id or uuid.uuid4().hex)
    token = (token or f"face_{profile_id[:8]}").strip()
    if not TOKEN_RE.fullmatch(token):
        raise ValueError("The face prompt token may contain letters, numbers, underscores and hyphens.")

    with _LOCK:
        record = {
            "id": profile_id,
            "name": name,
            "token": token,
            "description": description,
            "created_at": _utc_now(),
            "local_model_ready": False,
            "runpod_model_ready": False,
            "local_training_state": "not_trained",
            "runpod_training_state": "not_trained",
            "local_training_error": None,
            "local_training_job_id": None,
            "runpod_training_error": None,
            "runpod_training_job_id": None,
        }
        if _database_enabled():
            _ensure_database_migrated()
            from studio_db import create_face_profile_record
            result = create_face_profile_record(record)
        else:
            registry = _load_registry()
            existing = _find_raw(registry, profile_id)
            if existing:
                return _public_record(existing)
            if any(item.get("name", "").casefold() == name.casefold() for item in registry["profiles"]):
                raise ValueError("A face profile with that name already exists. Choose a different name.")
            if any(item.get("token") == token for item in registry["profiles"]):
                raise ValueError("That prompt token is already assigned to another profile. Choose a unique token.")
            registry["profiles"].append(record)
            _save_registry(registry)
            result = record
        (PROFILES_DIR / profile_id / "dataset").mkdir(parents=True, exist_ok=True)
        return _public_record(result)


def update_profile(profile_id: str, **fields) -> dict:
    profile_id = validate_profile_id(profile_id)
    allowed = {
        "local_model_ready",
        "runpod_model_ready",
        "local_training_state",
        "runpod_training_state",
        "local_training_error",
        "local_training_job_id",
        "runpod_training_error",
        "runpod_training_job_id",
        "token",
        "description",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unsupported profile fields: {', '.join(sorted(unknown))}")
    if "token" in fields and not TOKEN_RE.fullmatch(str(fields["token"] or "")):
        raise ValueError("Invalid face prompt token.")
    if "description" in fields:
        fields["description"] = validate_profile_description(fields["description"])
    with _LOCK:
        if _database_enabled():
            _ensure_database_migrated()
            from studio_db import update_face_profile_record
            update_face_profile_record(profile_id, fields)
            result = get_profile(profile_id)
            if result is None:
                raise ValueError("Face profile not found.")
            return result
        registry = _load_registry()
        record = _find_raw(registry, profile_id)
        if not record:
            raise ValueError("Face profile not found.")
        record.update(fields)
        _save_registry(registry)
        return _public_record(record)


def delete_profile(profile_id: str) -> bool:
    profile_id = validate_profile_id(profile_id)
    with _LOCK:
        if _database_enabled():
            _ensure_database_migrated()
            from studio_db import delete_face_profile_record
            deleted = delete_face_profile_record(profile_id)
        else:
            registry = _load_registry()
            record = _find_raw(registry, profile_id)
            if not record:
                return False
            registry["profiles"] = [item for item in registry["profiles"] if item.get("id") != profile_id]
            _save_registry(registry)
            deleted = True
        if deleted:
            rmtree(PROFILES_DIR / profile_id, ignore_errors=True)
            rmtree(LORA_DIR / profile_id, ignore_errors=True)
        return deleted


def dataset_dir(profile_id: str) -> Path:
    profile_id = validate_profile_id(profile_id)
    path = PROFILES_DIR / profile_id / "dataset"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_images(profile_id: str) -> list[Path]:
    folder = dataset_dir(profile_id)
    return sorted(
        item for item in folder.iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def save_profile_image(
    profile_id: str, image, *, group_id: str | None = None,
    angle_label: str | None = None, photo_kind: str = "manual",
    validation_status: str = "not_checked", validation_message: str | None = None,
    target_yaw: float | None = None, target_pitch: float | None = None,
    actual_yaw: float | None = None, actual_pitch: float | None = None,
    training_caption: str | None = None,
) -> Path:
    profile_id = validate_profile_id(profile_id)
    if not get_profile(profile_id):
        raise ValueError("Face profile not found.")
    folder = dataset_dir(profile_id)
    filename = f"face_{uuid.uuid4().hex}.jpg"
    destination = folder / filename
    # Re-encode so original filenames and EXIF metadata are not retained.
    image.convert("RGB").save(destination, format="JPEG", quality=95, optimize=True)
    # Grouped training sets are recorded in the database even in local development
    # mode, where the legacy ungrouped profile registry still lives in JSON.
    if _database_enabled() or group_id:
        from studio_db import add_face_photo
        add_face_photo(
            profile_id, destination.name, destination.read_bytes(), "image/jpeg",
            group_id=group_id, angle_label=angle_label, photo_kind=photo_kind,
            validation_status=validation_status, validation_message=validation_message,
            training_caption=(training_caption or angle_label or ""),
            target_yaw=target_yaw, target_pitch=target_pitch,
            actual_yaw=actual_yaw, actual_pitch=actual_pitch,
        )
    # Training captions let the LoRA associate a pose/expression/body description
    # with the shared trigger token instead of treating every image as identical.
    caption_path = destination.with_name(destination.name + ".caption.txt")
    caption = (training_caption or angle_label or "").strip()
    if caption and photo_kind not in {"source", "manual"}:
        caption_path.write_text(caption[:1200], encoding="utf-8")
    else:
        caption_path.unlink(missing_ok=True)
    return destination


def model_path(profile_id: str) -> Path:
    return LORA_DIR / validate_profile_id(profile_id)


def profile_ready_for_backend(profile_id: str, backend: str) -> bool:
    profile = get_profile(profile_id)
    if not profile:
        return False
    if backend == "runpod":
        return bool(profile.get("runpod_model_ready"))
    if backend == "local":
        return bool(profile.get("local_model_ready"))
    return False

def ensure_global_body_profile() -> dict:
    """Create or return the hidden, shared profile used only for global body-type training."""
    profile = get_profile(GLOBAL_BODY_PROFILE_ID)
    if profile:
        if profile.get("name") != GLOBAL_BODY_PROFILE_NAME:
            raise RuntimeError("The reserved shared-body profile ID is already used by another profile.")
        return profile
    return create_profile(
        GLOBAL_BODY_PROFILE_NAME,
        profile_id=GLOBAL_BODY_PROFILE_ID,
        token=GLOBAL_BODY_PROFILE_TOKEN,
        description=(
            "Shared body silhouette and physique conditioning used across all face identities. "
            "Train with captions describing adult body proportions, build, pose and clothing."
        ),
    )


def is_global_body_profile(profile_id: str | None) -> bool:
    return str(profile_id or "").strip().lower() == GLOBAL_BODY_PROFILE_ID
