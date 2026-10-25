"""PostgreSQL-backed metadata and media storage.

DATABASE_URL is loaded from the application's .env before this module is imported.
When it is absent, a local SQLite database is used for development only.
"""
from __future__ import annotations

import os
import threading
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint, create_engine, event, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

ROOT = Path(__file__).resolve().parent
_SCHEMA_LOCK = threading.RLock()
_SCHEMA_READY = set()


def _normalise_url(value: str) -> str:
    url = (value or "").strip()
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    elif url.startswith("postgresql+psycopg2://"):
        url = "postgresql+psycopg://" + url[len("postgresql+psycopg2://"):]
    return url


def database_url_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


def current_database_url() -> str:
    configured = _normalise_url(os.environ.get("DATABASE_URL", ""))
    if configured:
        return configured
    local_path = ROOT / "data" / "private_face_studio.db"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + local_path.as_posix()


class Base(DeclarativeBase):
    pass


class FaceProfileRow(Base):
    __tablename__ = "face_profiles"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    name_key: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    token: Mapped[str] = mapped_column(String(48), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    local_model_ready: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    runpod_model_ready: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    local_training_state: Mapped[str] = mapped_column(String(32), default="not_trained", nullable=False)
    runpod_training_state: Mapped[str] = mapped_column(String(32), default="not_trained", nullable=False)
    local_training_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_training_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    runpod_training_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    runpod_training_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


class FaceDatasetGroupRow(Base):
    __tablename__ = "face_dataset_groups"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    profile_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("face_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="generating")
    expected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    generated_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class FacePhotoRow(Base):
    __tablename__ = "face_photos"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    profile_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("face_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), default="image/jpeg", nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    image_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    group_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    angle_label: Mapped[str | None] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


class FaceModelArtifactRow(Base):
    __tablename__ = "face_model_artifacts"
    __table_args__ = (UniqueConstraint("profile_id", "backend", name="uq_face_model_profile_backend"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    profile_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("face_profiles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    backend: Mapped[str] = mapped_column(String(16), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_count: Mapped[int] = mapped_column(Integer, nullable=False)
    bundle_size: Mapped[int] = mapped_column(Integer, nullable=False)
    bundle_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class GeneratedAssetRow(Base):
    __tablename__ = "generated_assets"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex)
    asset_type: Mapped[str] = mapped_column(String(12), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    negative_prompt: Mapped[str] = mapped_column(Text, nullable=False, default="")
    profile_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    profile_name: Mapped[str | None] = mapped_column(String(80), nullable=True)
    backend: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")
    generation_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    media_size: Mapped[int] = mapped_column(Integer, nullable=False)
    media_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


@lru_cache(maxsize=4)
def _engine_for_url(url: str):
    if url.startswith("sqlite:"):
        engine = create_engine(url, connect_args={"check_same_thread": False})
        @event.listens_for(engine, "connect")
        def _sqlite_foreign_keys(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        return engine
    return create_engine(url, pool_pre_ping=True, pool_recycle=300)


def _session_factory():
    return sessionmaker(bind=_engine_for_url(current_database_url()), expire_on_commit=False)


def ensure_schema() -> None:
    url = current_database_url()
    with _SCHEMA_LOCK:
        if url in _SCHEMA_READY:
            return
        engine = _engine_for_url(url)
        Base.metadata.create_all(engine)
        # create_all does not add columns to a table created by an earlier version.
        # Add this column in-place so existing SQLite and PostgreSQL databases keep working.
        profile_columns = {column["name"] for column in inspect(engine).get_columns("face_profiles")}
        with engine.begin() as connection:
            if "description" not in profile_columns:
                connection.execute(text("ALTER TABLE face_profiles ADD COLUMN description TEXT NOT NULL DEFAULT ''"))
        photo_columns = {column["name"] for column in inspect(engine).get_columns("face_photos")}
        with engine.begin() as connection:
            if "group_id" not in photo_columns:
                connection.execute(text("ALTER TABLE face_photos ADD COLUMN group_id VARCHAR(32)"))
            if "angle_label" not in photo_columns:
                connection.execute(text("ALTER TABLE face_photos ADD COLUMN angle_label VARCHAR(80)"))
        _SCHEMA_READY.add(url)


def database_status() -> dict:
    url = current_database_url()
    try:
        ensure_schema()
        with _engine_for_url(url).connect() as connection:
            connection.execute(select(1))
        return {
            "connected": True,
            "engine": "postgresql" if url.startswith("postgresql") else "sqlite",
            "configured_postgresql": url.startswith("postgresql"),
            "message": "PostgreSQL connected." if url.startswith("postgresql") else
                ("Connected to an explicitly configured SQLite database." if database_url_configured()
                 else "DATABASE_URL is not set; using the local SQLite development database."),
        }
    except Exception:
        # Do not return SQLAlchemy exceptions here: they may contain connection details.
        return {
            "connected": False,
            "engine": "postgresql" if url.startswith("postgresql") else "sqlite",
            "configured_postgresql": url.startswith("postgresql"),
            "message": "Database connection failed. Check DATABASE_URL and the database network/access settings in .env.",
        }


def _profile_dict(row: FaceProfileRow) -> dict:
    return {
        "id": row.id, "name": row.name, "token": row.token, "description": row.description or "", "created_at": row.created_at,
        "local_model_ready": bool(row.local_model_ready), "runpod_model_ready": bool(row.runpod_model_ready),
        "local_training_state": row.local_training_state, "runpod_training_state": row.runpod_training_state,
        "local_training_error": row.local_training_error, "local_training_job_id": row.local_training_job_id,
        "runpod_training_error": row.runpod_training_error, "runpod_training_job_id": row.runpod_training_job_id,
    }


def get_face_profile_record(profile_id: str) -> dict | None:
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(FaceProfileRow, profile_id)
        if not row:
            return None
        result = _profile_dict(row)
        result["image_count"] = session.scalar(
            select(func.count()).select_from(FacePhotoRow).where(FacePhotoRow.profile_id == profile_id)
        ) or 0
        return result


def list_face_profile_records() -> list[dict]:
    ensure_schema()
    with _session_factory()() as session:
        rows = session.scalars(select(FaceProfileRow).order_by(FaceProfileRow.created_at.asc())).all()
        result = []
        for row in rows:
            record = _profile_dict(row)
            record["image_count"] = session.scalar(
                select(func.count()).select_from(FacePhotoRow).where(FacePhotoRow.profile_id == row.id)
            ) or 0
            result.append(record)
        return result


def insert_face_profile_record(record: dict) -> dict:
    ensure_schema()
    values = {
        "id": record["id"], "name": record["name"], "name_key": record["name"].casefold(),
        "token": record["token"], "description": record.get("description", "") or "", "created_at": record["created_at"],
        "local_model_ready": bool(record.get("local_model_ready", False)),
        "runpod_model_ready": bool(record.get("runpod_model_ready", False)),
        "local_training_state": record.get("local_training_state", "not_trained"),
        "runpod_training_state": record.get("runpod_training_state", "not_trained"),
        "local_training_error": record.get("local_training_error"),
        "local_training_job_id": record.get("local_training_job_id"),
        "runpod_training_error": record.get("runpod_training_error"),
        "runpod_training_job_id": record.get("runpod_training_job_id"),
    }
    with _session_factory()() as session:
        try:
            row = session.get(FaceProfileRow, values["id"])
            if row:
                return _profile_dict(row)
            row = FaceProfileRow(**values)
            session.add(row)
            session.commit()
            return _profile_dict(row)
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("A face profile name or prompt token already exists in PostgreSQL.") from exc


def create_face_profile_record(record: dict) -> dict:
    ensure_schema()
    with _session_factory()() as session:
        try:
            existing = session.get(FaceProfileRow, record["id"])
            if existing:
                return _profile_dict(existing)
            row = FaceProfileRow(
                id=record["id"], name=record["name"], name_key=record["name"].casefold(),
                token=record["token"], description=record.get("description", "") or "", created_at=record["created_at"],
                local_model_ready=False, runpod_model_ready=False,
                local_training_state="not_trained", runpod_training_state="not_trained",
                local_training_error=None, local_training_job_id=None,
                runpod_training_error=None, runpod_training_job_id=None,
            )
            session.add(row)
            session.commit()
            return _profile_dict(row)
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("A face profile with that name or prompt token already exists. Choose a unique value.") from exc


def update_face_profile_record(profile_id: str, fields: dict) -> dict:
    ensure_schema()
    allowed = {
        "local_model_ready", "runpod_model_ready", "local_training_state", "runpod_training_state",
        "local_training_error", "local_training_job_id", "runpod_training_error", "runpod_training_job_id", "token", "description",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError("Unsupported profile fields: " + ", ".join(sorted(unknown)))
    with _session_factory()() as session:
        row = session.get(FaceProfileRow, profile_id)
        if not row:
            raise ValueError("Face profile not found.")
        for key, value in fields.items():
            setattr(row, key, value)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("That prompt token is already assigned to another profile.") from exc
        return _profile_dict(row)


def delete_face_profile_record(profile_id: str) -> bool:
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(FaceProfileRow, profile_id)
        if not row:
            return False
        session.delete(row)
        session.commit()
        return True


def add_face_photo(
    profile_id: str, filename: str, image_data: bytes, mime_type: str = "image/jpeg",
    group_id: str | None = None, angle_label: str | None = None,
) -> str:
    ensure_schema()
    with _session_factory()() as session:
        if not session.get(FaceProfileRow, profile_id):
            raise ValueError("Face profile is not present in the configured database.")
        if group_id:
            group = session.get(FaceDatasetGroupRow, group_id)
            if not group or group.profile_id != profile_id:
                raise ValueError("Dataset group was not found for this face profile.")
        row = FacePhotoRow(
            id=uuid.uuid4().hex, profile_id=profile_id, filename=filename, mime_type=mime_type,
            byte_size=len(image_data), image_data=image_data, group_id=group_id,
            angle_label=(angle_label or "")[:80] or None,
        )
        session.add(row)
        session.commit()
        return row.id


def _dataset_group_dict(row: FaceDatasetGroupRow) -> dict:
    return {
        "id": row.id, "profile_id": row.profile_id, "title": row.title,
        "status": row.status, "expected_count": row.expected_count,
        "generated_count": row.generated_count, "error": row.error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def create_face_dataset_group(profile_id: str, title: str, expected_count: int = 5) -> dict:
    title = (title or "").strip()
    if not title or len(title) > 120:
        raise ValueError("Dataset group title must contain 1–120 characters.")
    if not 1 <= int(expected_count) <= 10:
        raise ValueError("Dataset group expected count must be between 1 and 10.")
    ensure_schema()
    with _session_factory()() as session:
        if not session.get(FaceProfileRow, profile_id):
            raise ValueError("Face profile is not present in the configured database.")
        row = FaceDatasetGroupRow(
            id=uuid.uuid4().hex, profile_id=profile_id, title=title,
            status="generating", expected_count=int(expected_count), generated_count=0,
        )
        session.add(row)
        session.commit()
        return _dataset_group_dict(row)


def get_face_dataset_group(group_id: str) -> dict | None:
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(FaceDatasetGroupRow, group_id)
        return _dataset_group_dict(row) if row else None


def update_face_dataset_group(
    group_id: str, *, title: str | None = None, status: str | None = None,
    generated_count: int | None = None, expected_count: int | None = None,
    error: str | None = None,
) -> dict | None:
    allowed_statuses = {"generating", "completed", "partial", "failed"}
    if status is not None and status not in allowed_statuses:
        raise ValueError("Invalid dataset group status.")
    normalized_title = None
    if title is not None:
        normalized_title = str(title).strip()
        if not normalized_title or len(normalized_title) > 120:
            raise ValueError("Dataset group name must contain 1–120 characters.")
    if expected_count is not None:
        expected_count = int(expected_count)
        if not 1 <= expected_count <= 30:
            raise ValueError("Expected photo count must be between 1 and 30.")
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(FaceDatasetGroupRow, group_id)
        if not row:
            return None
        if normalized_title is not None:
            row.title = normalized_title
        if expected_count is not None:
            row.expected_count = max(expected_count, int(generated_count or row.generated_count or 0))
        if generated_count is not None:
            count = int(generated_count)
            if count < 0 or count > max(row.expected_count, int(expected_count or 0)):
                raise ValueError("Generated count is outside the expected range.")
            row.generated_count = count
        if status is not None:
            row.status = status
        if error is not None:
            row.error = str(error)[:1200] or None
        row.updated_at = datetime.now(timezone.utc)
        session.commit()
        return _dataset_group_dict(row)


def list_face_dataset_groups(profile_id: str | None = None, limit: int = 200) -> list[dict]:
    ensure_schema()
    with _session_factory()() as session:
        query = select(FaceDatasetGroupRow)
        if profile_id:
            query = query.where(FaceDatasetGroupRow.profile_id == profile_id)
        rows = session.scalars(
            query.order_by(FaceDatasetGroupRow.created_at.desc()).limit(max(1, min(int(limit), 500)))
        ).all()
        result = []
        for row in rows:
            record = _dataset_group_dict(row)
            record["profile_name"] = (
                session.get(FaceProfileRow, row.profile_id).name
                if session.get(FaceProfileRow, row.profile_id) else "Unknown profile"
            )
            photos = session.scalars(
                select(FacePhotoRow).where(FacePhotoRow.group_id == row.id)
                .order_by(FacePhotoRow.created_at.asc(), FacePhotoRow.id.asc())
            ).all()
            record["photos"] = [
                {
                    "id": photo.id, "filename": photo.filename,
                    "angle_label": photo.angle_label or "Training photo",
                    "mime_type": photo.mime_type, "byte_size": photo.byte_size,
                    "created_at": photo.created_at.isoformat() if photo.created_at else None,
                }
                for photo in photos
            ]
            record["photo_count"] = len(photos)
            result.append(record)
        return result


def delete_face_dataset_group(group_id: str) -> dict | None:
    """Delete a group's metadata and its grouped photo records in one transaction."""
    ensure_schema()
    with _session_factory()() as session:
        group = session.get(FaceDatasetGroupRow, group_id)
        if not group:
            return None
        photos = session.scalars(
            select(FacePhotoRow).where(FacePhotoRow.group_id == group_id)
        ).all()
        result = _dataset_group_dict(group)
        result["photos"] = [
            {"id": photo.id, "profile_id": photo.profile_id, "filename": photo.filename}
            for photo in photos
        ]
        for photo in photos:
            session.delete(photo)
        session.delete(group)
        session.commit()
        return result


def list_face_photo_names(profile_id: str) -> set[str]:
    ensure_schema()
    with _session_factory()() as session:
        rows = session.scalars(select(FacePhotoRow.filename).where(FacePhotoRow.profile_id == profile_id)).all()
        return set(rows)


def list_face_photo_records(profile_id: str, filenames: set[str] | None = None) -> list[dict]:
    """Return stored photo names and bytes for rebuilding a profile's local training cache."""
    ensure_schema()
    with _session_factory()() as session:
        query = select(FacePhotoRow).where(FacePhotoRow.profile_id == profile_id)
        if filenames is not None:
            if not filenames:
                return []
            query = query.where(FacePhotoRow.filename.in_(filenames))
        rows = session.scalars(query.order_by(FacePhotoRow.created_at.asc())).all()
        return [
            {
                "id": row.id, "profile_id": row.profile_id, "filename": row.filename,
                "mime_type": row.mime_type, "byte_size": row.byte_size,
                "group_id": row.group_id, "angle_label": row.angle_label,
                "image_data": bytes(row.image_data),
            }
            for row in rows
        ]


def get_face_photo_bytes(photo_id: str) -> tuple[dict, bytes] | None:
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(FacePhotoRow, photo_id)
        if not row:
            return None
        return ({"id": row.id, "profile_id": row.profile_id, "filename": row.filename,
                 "mime_type": row.mime_type, "byte_size": row.byte_size,
                 "group_id": row.group_id, "angle_label": row.angle_label}, bytes(row.image_data))


def update_face_photo(
    photo_id: str, *, angle_label: str | None = None,
    image_data: bytes | None = None, mime_type: str | None = None,
) -> dict | None:
    """Update a grouped training photo's label and/or encoded image bytes."""
    ensure_schema()
    if image_data is not None and (not image_data or len(image_data) > 8 * 1024 * 1024):
        raise ValueError("Replacement photos must be non-empty and 8 MB or smaller.")
    normalized_label = None
    if angle_label is not None:
        normalized_label = str(angle_label).strip()
        if not normalized_label or len(normalized_label) > 80:
            raise ValueError("Photo label must contain 1–80 characters.")
    with _session_factory()() as session:
        row = session.get(FacePhotoRow, photo_id)
        if not row:
            return None
        if normalized_label is not None:
            row.angle_label = normalized_label
        if image_data is not None:
            row.image_data = image_data
            row.byte_size = len(image_data)
            row.mime_type = (mime_type or "image/jpeg")[:100]
        session.commit()
        return {
            "id": row.id, "profile_id": row.profile_id, "filename": row.filename,
            "group_id": row.group_id, "angle_label": row.angle_label or "Training photo",
            "mime_type": row.mime_type, "byte_size": row.byte_size,
        }


def delete_face_photo(photo_id: str) -> dict | None:
    """Delete one photo record and return local-file metadata for cleanup."""
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(FacePhotoRow, photo_id)
        if not row:
            return None
        result = {
            "id": row.id, "profile_id": row.profile_id, "filename": row.filename,
            "group_id": row.group_id, "angle_label": row.angle_label,
        }
        session.delete(row)
        session.commit()
        return result


def save_face_model_bundle(profile_id: str, backend: str, filename: str, bundle_data: bytes, file_count: int) -> dict:
    if backend not in {"local", "runpod"}:
        raise ValueError("Face model backend must be local or runpod.")
    if not bundle_data or len(bundle_data) > 512 * 1024 * 1024:
        raise ValueError("Face model bundle must be non-empty and no larger than 512 MB.")
    ensure_schema()
    with _session_factory()() as session:
        if not session.get(FaceProfileRow, profile_id):
            raise ValueError("Face profile is not present in the configured database.")
        row = session.scalars(
            select(FaceModelArtifactRow).where(
                FaceModelArtifactRow.profile_id == profile_id,
                FaceModelArtifactRow.backend == backend,
            )
        ).first()
        if row is None:
            row = FaceModelArtifactRow(
                id=uuid.uuid4().hex, profile_id=profile_id, backend=backend,
                filename=Path(filename).name, file_count=file_count,
                bundle_size=len(bundle_data), bundle_data=bundle_data,
            )
            session.add(row)
        else:
            row.filename = Path(filename).name
            row.file_count = file_count
            row.bundle_size = len(bundle_data)
            row.bundle_data = bundle_data
            row.updated_at = datetime.now(timezone.utc)
        session.commit()
        return {
            "id": row.id, "profile_id": row.profile_id, "backend": row.backend,
            "filename": row.filename, "file_count": row.file_count, "bundle_size": row.bundle_size,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }


def get_face_model_bundle(profile_id: str, backend: str) -> dict | None:
    ensure_schema()
    with _session_factory()() as session:
        row = session.scalars(
            select(FaceModelArtifactRow).where(
                FaceModelArtifactRow.profile_id == profile_id,
                FaceModelArtifactRow.backend == backend,
            )
        ).first()
        if not row:
            return None
        return {
            "id": row.id, "profile_id": row.profile_id, "backend": row.backend,
            "filename": row.filename, "file_count": row.file_count, "bundle_size": row.bundle_size,
            "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            "bundle_data": bytes(row.bundle_data),
        }


def _asset_dict(row: GeneratedAssetRow, include_content: bool = False) -> dict:
    result = {
        "id": row.id, "asset_type": row.asset_type, "title": row.title, "filename": row.filename,
        "mime_type": row.mime_type, "prompt": row.prompt, "negative_prompt": row.negative_prompt,
        "profile_id": row.profile_id, "profile_name": row.profile_name, "backend": row.backend,
        "generation_mode": row.generation_mode, "media_size": row.media_size,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    if include_content:
        result["media_data"] = bytes(row.media_data)
    return result


def create_asset(*, asset_type: str, title: str, filename: str, mime_type: str, media_data: bytes,
                 prompt: str = "", negative_prompt: str = "", profile_id: str | None = None,
                 profile_name: str | None = None, backend: str = "cpu", generation_mode: str = "") -> dict:
    if asset_type not in {"image", "video"}:
        raise ValueError("Asset type must be image or video.")
    if not media_data:
        raise ValueError("Cannot save an empty media file.")
    if len(media_data) > 512 * 1024 * 1024:
        raise ValueError("Media files must be 512 MB or smaller to store in the database.")
    ensure_schema()
    with _session_factory()() as session:
        row = GeneratedAssetRow(
            id=uuid.uuid4().hex, asset_type=asset_type, title=(title or "").strip()[:160] or
            ("Generated image" if asset_type == "image" else "Generated video"),
            filename=Path(filename).name, mime_type=mime_type, prompt=prompt[:1200],
            negative_prompt=negative_prompt[:600], profile_id=profile_id, profile_name=profile_name,
            backend=backend[:32], generation_mode=generation_mode[:32], media_size=len(media_data),
            media_data=media_data,
        )
        session.add(row)
        session.commit()
        return _asset_dict(row)


def list_asset_filenames(asset_type: str) -> set[str]:
    if asset_type not in {"image", "video"}:
        raise ValueError("Asset type must be image or video.")
    ensure_schema()
    with _session_factory()() as session:
        filenames = session.scalars(
            select(GeneratedAssetRow.filename).where(GeneratedAssetRow.asset_type == asset_type)
        ).all()
        return {Path(name).name for name in filenames}


def _asset_query(asset_type: str | None = None, search: str = ""):
    query = select(GeneratedAssetRow)
    if asset_type in {"image", "video"}:
        query = query.where(GeneratedAssetRow.asset_type == asset_type)
    term = (search or "").strip()[:160]
    if term:
        pattern = f"%{term}%"
        query = query.where(
            GeneratedAssetRow.title.ilike(pattern)
            | GeneratedAssetRow.prompt.ilike(pattern)
            | GeneratedAssetRow.negative_prompt.ilike(pattern)
            | GeneratedAssetRow.profile_name.ilike(pattern)
        )
    return query


def count_assets(asset_type: str | None = None, search: str = "") -> int:
    ensure_schema()
    with _session_factory()() as session:
        query = _asset_query(asset_type, search).with_only_columns(func.count(GeneratedAssetRow.id)).order_by(None)
        return int(session.scalar(query) or 0)


def list_assets(
    asset_type: str | None = None,
    limit: int = 200,
    offset: int = 0,
    search: str = "",
    sort: str = "newest",
) -> list[dict]:
    ensure_schema()
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    with _session_factory()() as session:
        query = _asset_query(asset_type, search)
        if sort == "oldest":
            ordering = (GeneratedAssetRow.created_at.asc(), GeneratedAssetRow.id.asc())
        elif sort == "title":
            ordering = (func.lower(GeneratedAssetRow.title).asc(), GeneratedAssetRow.created_at.desc())
        else:
            ordering = (GeneratedAssetRow.created_at.desc(), GeneratedAssetRow.id.desc())
        rows = session.scalars(query.order_by(*ordering).offset(offset).limit(limit)).all()
        return [_asset_dict(row) for row in rows]


def get_asset(asset_id: str, include_content: bool = False) -> dict | None:
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(GeneratedAssetRow, asset_id)
        return _asset_dict(row, include_content=include_content) if row else None


def update_asset(asset_id: str, fields: dict) -> dict | None:
    allowed = {"title", "prompt", "negative_prompt"}
    if not fields or set(fields) - allowed:
        raise ValueError("Update title, prompt, and/or negative prompt only.")
    if "title" in fields:
        title = str(fields["title"] or "").strip()
        if not title or len(title) > 160:
            raise ValueError("Title must contain 1–160 characters.")
        fields["title"] = title
    if "prompt" in fields and len(str(fields["prompt"] or "")) > 1200:
        raise ValueError("Prompt must be 1200 characters or fewer.")
    if "negative_prompt" in fields and len(str(fields["negative_prompt"] or "")) > 600:
        raise ValueError("Negative prompt must be 600 characters or fewer.")
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(GeneratedAssetRow, asset_id)
        if not row:
            return None
        for key, value in fields.items():
            setattr(row, key, value)
        session.commit()
        return _asset_dict(row)


def delete_asset(asset_id: str) -> dict | None:
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(GeneratedAssetRow, asset_id)
        if not row:
            return None
        result = _asset_dict(row)
        session.delete(row)
        session.commit()
        return result
