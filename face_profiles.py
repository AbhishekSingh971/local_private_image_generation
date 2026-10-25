"""Persistent named face profiles for local and RunPod-backed LoRA workflows.

Profile metadata and face-photo bytes use PostgreSQL whenever DATABASE_URL is configured.
A one-time migration imports the previous JSON registry and any existing profile images.
"""
from __future__ import annotations

import json
import os
import re
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PROFILE_ID_RE = re.compile(r"^[a-f0-9]{32}$")
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{1,48}$")
_LOCK = threading.RLock()
_DB_MIGRATED = False

for folder in (DATA_DIR, PROFILES_DIR, LORA_DIR):
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


def _public_record(record: dict) -> dict:
    profile_id = record["id"]
    result = dict(record)
    result["image_count"] = max(_image_count(profile_id), int(record.get("image_count", 0) or 0))
    result["local_model_ready"] = _local_lora_ready(profile_id)
    result["runpod_model_ready"] = bool(record.get("runpod_model_ready"))
    result.pop("dataset_dir", None)
    return result


def _ensure_database_migrated() -> None:
    """Import profiles from the old JSON registry once, preserving existing datasets."""
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
        _DB_MIGRATED = True


def list_profiles() -> list[dict]:
    if _database_enabled():
        _ensure_database_migrated()
        from studio_db import list_face_profile_records
        records = list_face_profile_records()
        return [_public_record(record) for record in records]
    with _LOCK:
        registry = _load_registry()
        return [
            _public_record(record)
            for record in sorted(registry["profiles"], key=lambda item: item.get("created_at", ""))
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


def create_profile(name: str, profile_id: str | None = None, token: str | None = None) -> dict:
    name = (name or "").strip()
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
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unsupported profile fields: {', '.join(sorted(unknown))}")
    if "token" in fields and not TOKEN_RE.fullmatch(str(fields["token"] or "")):
        raise ValueError("Invalid face prompt token.")
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


def save_profile_image(profile_id: str, image) -> Path:
    profile_id = validate_profile_id(profile_id)
    if not get_profile(profile_id):
        raise ValueError("Face profile not found.")
    folder = dataset_dir(profile_id)
    filename = f"face_{uuid.uuid4().hex}.jpg"
    destination = folder / filename
    # Re-encode so original filenames and EXIF metadata are not retained.
    image.convert("RGB").save(destination, format="JPEG", quality=95, optimize=True)
    if _database_enabled():
        from studio_db import add_face_photo
        add_face_photo(profile_id, destination.name, destination.read_bytes(), "image/jpeg")
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
