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

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, LargeBinary, String, Text, create_engine, event, func, select
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
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    local_model_ready: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    runpod_model_ready: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    local_training_state: Mapped[str] = mapped_column(String(32), default="not_trained", nullable=False)
    runpod_training_state: Mapped[str] = mapped_column(String(32), default="not_trained", nullable=False)
    local_training_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    local_training_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    runpod_training_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    runpod_training_job_id: Mapped[str | None] = mapped_column(String(128), nullable=True)


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
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)


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
        Base.metadata.create_all(_engine_for_url(url))
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
            "configured_postgresql": database_url_configured(),
            "message": "PostgreSQL connected." if url.startswith("postgresql") else
                ("Connected to an explicitly configured SQLite database." if database_url_configured()
                 else "DATABASE_URL is not set; using the local SQLite development database."),
        }
    except Exception:
        # Do not return SQLAlchemy exceptions here: they may contain connection details.
        return {
            "connected": False,
            "engine": "postgresql" if url.startswith("postgresql") else "sqlite",
            "configured_postgresql": database_url_configured(),
            "message": "Database connection failed. Check DATABASE_URL and the database network/access settings in .env.",
        }


def _profile_dict(row: FaceProfileRow) -> dict:
    return {
        "id": row.id, "name": row.name, "token": row.token, "created_at": row.created_at,
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
        "token": record["token"], "created_at": record["created_at"],
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
            row = FaceProfileRow(
                id=record["id"], name=record["name"], name_key=record["name"].casefold(),
                token=record["token"], created_at=record["created_at"],
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
        "local_training_error", "local_training_job_id", "runpod_training_error", "runpod_training_job_id", "token",
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


def add_face_photo(profile_id: str, filename: str, image_data: bytes, mime_type: str = "image/jpeg") -> None:
    ensure_schema()
    with _session_factory()() as session:
        if not session.get(FaceProfileRow, profile_id):
            raise ValueError("Face profile is not present in the configured database.")
        session.add(FacePhotoRow(
            id=uuid.uuid4().hex, profile_id=profile_id, filename=filename, mime_type=mime_type,
            byte_size=len(image_data), image_data=image_data,
        ))
        session.commit()


def list_face_photo_names(profile_id: str) -> set[str]:
    ensure_schema()
    with _session_factory()() as session:
        rows = session.scalars(select(FacePhotoRow.filename).where(FacePhotoRow.profile_id == profile_id)).all()
        return set(rows)


def get_face_photo_bytes(photo_id: str) -> tuple[dict, bytes] | None:
    ensure_schema()
    with _session_factory()() as session:
        row = session.get(FacePhotoRow, photo_id)
        if not row:
            return None
        return ({"id": row.id, "profile_id": row.profile_id, "filename": row.filename,
                 "mime_type": row.mime_type, "byte_size": row.byte_size}, bytes(row.image_data))


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


def list_assets(asset_type: str | None = None, limit: int = 200) -> list[dict]:
    ensure_schema()
    limit = max(1, min(int(limit), 500))
    with _session_factory()() as session:
        query = select(GeneratedAssetRow)
        if asset_type in {"image", "video"}:
            query = query.where(GeneratedAssetRow.asset_type == asset_type)
        rows = session.scalars(query.order_by(GeneratedAssetRow.created_at.desc()).limit(limit)).all()
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
