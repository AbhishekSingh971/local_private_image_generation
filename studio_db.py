"""Supabase PostgreSQL storage using psycopg directly (no SQLAlchemy)."""
from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg.rows import dict_row
from psycopg import sql
from psycopg.errors import UniqueViolation

ROOT = Path(__file__).resolve().parent
_SCHEMA_LOCK = threading.RLock()
_SCHEMA_READY: set[str] = set()


def database_url_configured() -> bool:
    return bool(os.environ.get("DATABASE_URL", "").strip())


def current_database_url() -> str:
    value = (os.environ.get("DATABASE_URL") or "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is required. Set it to your Supabase PostgreSQL connection URI in .env.")
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    if value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value[len("postgresql+psycopg://"):]
    if value.startswith("postgresql+psycopg2://"):
        value = "postgresql://" + value[len("postgresql+psycopg2://"):]
    if not value.startswith("postgresql://"):
        raise RuntimeError("DATABASE_URL must be a PostgreSQL URI from Supabase. SQLite is not supported.")
    return value


def _connect():
    return psycopg.connect(current_database_url(), row_factory=dict_row, connect_timeout=10)


def _now():
    return datetime.now(timezone.utc)


def _iso(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _one(query, params=()):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone()


def _all(query, params=()):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchall()


def _run(query, params=(), returning=False):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(query, params)
        return cur.fetchone() if returning else cur.rowcount


def _ensure_column(conn, table, name, definition):
    with conn.cursor() as cur:
        cur.execute(sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} ").format(
            sql.Identifier(table), sql.Identifier(name)) + sql.SQL(definition))


def ensure_schema() -> None:
    url = current_database_url()
    with _SCHEMA_LOCK:
        if url in _SCHEMA_READY:
            return
        statements = [
            """CREATE TABLE IF NOT EXISTS face_profiles (
                id VARCHAR(32) PRIMARY KEY, name VARCHAR(80) NOT NULL,
                name_key VARCHAR(80) UNIQUE NOT NULL, token VARCHAR(48) UNIQUE NOT NULL,
                description TEXT NOT NULL DEFAULT '', created_at VARCHAR(40) NOT NULL,
                local_model_ready BOOLEAN NOT NULL DEFAULT FALSE, runpod_model_ready BOOLEAN NOT NULL DEFAULT FALSE,
                local_training_state VARCHAR(32) NOT NULL DEFAULT 'not_trained',
                runpod_training_state VARCHAR(32) NOT NULL DEFAULT 'not_trained',
                local_training_error TEXT, local_training_job_id VARCHAR(128),
                runpod_training_error TEXT, runpod_training_job_id VARCHAR(128))""",
            """CREATE TABLE IF NOT EXISTS face_dataset_groups (
                id VARCHAR(32) PRIMARY KEY, profile_id VARCHAR(32) NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                title VARCHAR(120) NOT NULL, dataset_type VARCHAR(24) NOT NULL DEFAULT 'face_angles',
                prompt TEXT NOT NULL DEFAULT '', status VARCHAR(24) NOT NULL DEFAULT 'generating',
                expected_count INTEGER NOT NULL DEFAULT 5, generated_count INTEGER NOT NULL DEFAULT 0,
                error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS face_photos (
                id VARCHAR(32) PRIMARY KEY, profile_id VARCHAR(32) NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                filename VARCHAR(255) NOT NULL, mime_type VARCHAR(100) NOT NULL DEFAULT 'image/jpeg',
                byte_size INTEGER NOT NULL, image_data BYTEA NOT NULL, group_id VARCHAR(32),
                angle_label VARCHAR(80), photo_kind VARCHAR(16) NOT NULL DEFAULT 'manual',
                validation_status VARCHAR(24) NOT NULL DEFAULT 'not_checked', validation_message TEXT,
                training_caption TEXT NOT NULL DEFAULT '', target_yaw DOUBLE PRECISION, target_pitch DOUBLE PRECISION,
                actual_yaw DOUBLE PRECISION, actual_pitch DOUBLE PRECISION, created_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS face_dataset_jobs (
                id VARCHAR(32) PRIMARY KEY, group_id VARCHAR(32) NOT NULL REFERENCES face_dataset_groups(id) ON DELETE CASCADE,
                profile_id VARCHAR(32) NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                dataset_type VARCHAR(24) NOT NULL, status VARCHAR(24) NOT NULL DEFAULT 'queued',
                backend VARCHAR(24) NOT NULL DEFAULT 'cpu', expected_count INTEGER NOT NULL,
                generated_count INTEGER NOT NULL DEFAULT 0, current_index INTEGER NOT NULL DEFAULT 0,
                current_label VARCHAR(80) NOT NULL DEFAULT '', current_task_id VARCHAR(32),
                source_photo_id VARCHAR(32) NOT NULL, group_count_base INTEGER NOT NULL DEFAULT 0,
                target_manifest TEXT NOT NULL DEFAULT '[]', cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
                failed_targets TEXT NOT NULL DEFAULT '[]', review_targets TEXT NOT NULL DEFAULT '[]',
                error TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS face_model_artifacts (
                id VARCHAR(32) PRIMARY KEY, profile_id VARCHAR(32) NOT NULL REFERENCES face_profiles(id) ON DELETE CASCADE,
                backend VARCHAR(16) NOT NULL, filename VARCHAR(255) NOT NULL, file_count INTEGER NOT NULL,
                bundle_size INTEGER NOT NULL, bundle_data BYTEA NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT uq_face_model_profile_backend UNIQUE(profile_id, backend))""",
            """CREATE TABLE IF NOT EXISTS generated_assets (
                id VARCHAR(32) PRIMARY KEY, asset_type VARCHAR(12) NOT NULL, title VARCHAR(160) NOT NULL,
                filename VARCHAR(255) NOT NULL, mime_type VARCHAR(100) NOT NULL, prompt TEXT NOT NULL DEFAULT '',
                negative_prompt TEXT NOT NULL DEFAULT '', profile_id VARCHAR(32), profile_name VARCHAR(80),
                backend VARCHAR(32) NOT NULL DEFAULT 'cpu', generation_mode VARCHAR(32) NOT NULL DEFAULT '',
                media_size INTEGER NOT NULL, media_data BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
        ]
        with psycopg.connect(url, row_factory=dict_row, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
            _ensure_column(conn, "face_profiles", "description", "TEXT NOT NULL DEFAULT ''")
            for name, definition in (("dataset_type", "VARCHAR(24) NOT NULL DEFAULT 'face_angles'"), ("prompt", "TEXT NOT NULL DEFAULT ''"), ("generated_count", "INTEGER NOT NULL DEFAULT 0")):
                _ensure_column(conn, "face_dataset_groups", name, definition)
            # Older installations may already have generated_count but no database
            # default. Background dataset jobs insert groups before any images exist,
            # so repair that schema in place and backfill any legacy NULLs if present.
            with conn.cursor() as cur:
                cur.execute("UPDATE face_dataset_groups SET generated_count = 0 WHERE generated_count IS NULL")
                cur.execute("ALTER TABLE face_dataset_groups ALTER COLUMN generated_count SET DEFAULT 0")
            for name, definition in (("group_count_base", "INTEGER NOT NULL DEFAULT 0"), ("target_manifest", "TEXT NOT NULL DEFAULT '[]'")):
                _ensure_column(conn, "face_dataset_jobs", name, definition)
            for name, definition in (
                ("group_id", "VARCHAR(32)"), ("angle_label", "VARCHAR(80)"),
                ("photo_kind", "VARCHAR(16) NOT NULL DEFAULT 'manual'"),
                ("validation_status", "VARCHAR(24) NOT NULL DEFAULT 'not_checked'"),
                ("validation_message", "TEXT"), ("training_caption", "TEXT NOT NULL DEFAULT ''"),
                ("target_yaw", "DOUBLE PRECISION"), ("target_pitch", "DOUBLE PRECISION"),
                ("actual_yaw", "DOUBLE PRECISION"), ("actual_pitch", "DOUBLE PRECISION")):
                _ensure_column(conn, "face_photos", name, definition)
            _ensure_column(conn, "face_photos", "created_at", "TIMESTAMPTZ NOT NULL DEFAULT now()")
            with conn.cursor() as cur:
                cur.execute("CREATE INDEX IF NOT EXISTS ix_face_dataset_groups_profile_id ON face_dataset_groups(profile_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_face_photos_profile_id ON face_photos(profile_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_face_photos_group_id ON face_photos(group_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_face_dataset_jobs_group_id ON face_dataset_jobs(group_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_face_dataset_jobs_profile_id ON face_dataset_jobs(profile_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_face_dataset_jobs_status ON face_dataset_jobs(status)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_face_model_artifacts_profile_id ON face_model_artifacts(profile_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_generated_assets_asset_type ON generated_assets(asset_type)")
                cur.execute("CREATE INDEX IF NOT EXISTS ix_generated_assets_profile_id ON generated_assets(profile_id)")
        _SCHEMA_READY.add(url)


def database_status() -> dict:
    try:
        ensure_schema()
        _one("SELECT 1")
        return {"connected": True, "engine": "postgresql", "configured_postgresql": True, "message": "Supabase PostgreSQL connected."}
    except Exception:
        return {"connected": False, "engine": "postgresql", "configured_postgresql": database_url_configured(),
                "message": "Database connection failed. Check DATABASE_URL and Supabase network/access settings in .env."}


def _profile_dict(row):
    if not row: return None
    return {k: row.get(k) for k in ("id","name","token","description","created_at","local_model_ready","runpod_model_ready",
        "local_training_state","runpod_training_state","local_training_error","local_training_job_id","runpod_training_error","runpod_training_job_id")}


def get_face_profile_record(profile_id):
    ensure_schema()
    row = _one("SELECT p.*, (SELECT count(*) FROM face_photos f WHERE f.profile_id=p.id) AS image_count FROM face_profiles p WHERE p.id=%s", (profile_id,))
    return dict(_profile_dict(row), image_count=int(row["image_count"])) if row else None


def list_face_profile_records():
    ensure_schema()
    rows = _all("SELECT p.*, (SELECT count(*) FROM face_photos f WHERE f.profile_id=p.id) AS image_count FROM face_profiles p ORDER BY p.created_at ASC")
    return [dict(_profile_dict(r), image_count=int(r["image_count"])) for r in rows]


def _profile_values(record):
    return (record["id"],record["name"],record["name"].casefold(),record["token"],record.get("description","") or "",record["created_at"],
        bool(record.get("local_model_ready",False)),bool(record.get("runpod_model_ready",False)),
        record.get("local_training_state","not_trained"),record.get("runpod_training_state","not_trained"),
        record.get("local_training_error"),record.get("local_training_job_id"),record.get("runpod_training_error"),record.get("runpod_training_job_id"))


def insert_face_profile_record(record):
    ensure_schema()
    values=_profile_values(record)
    try:
        _run("""INSERT INTO face_profiles(id,name,name_key,token,description,created_at,local_model_ready,runpod_model_ready,
            local_training_state,runpod_training_state,local_training_error,local_training_job_id,runpod_training_error,runpod_training_job_id)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING""", values)
    except UniqueViolation as exc:
        raise ValueError("A face profile name or prompt token already exists in PostgreSQL.") from exc
    return get_face_profile_record(record["id"])


def create_face_profile_record(record):
    ensure_schema()
    values=_profile_values(record)
    try:
        _run("""INSERT INTO face_profiles(id,name,name_key,token,description,created_at,local_model_ready,runpod_model_ready,
            local_training_state,runpod_training_state,local_training_error,local_training_job_id,runpod_training_error,runpod_training_job_id)
            VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(id) DO NOTHING""", values)
    except UniqueViolation as exc:
        raise ValueError("A face profile with that name or prompt token already exists. Choose a unique value.") from exc
    return get_face_profile_record(record["id"])


def update_face_profile_record(profile_id, fields):
    allowed={"local_model_ready","runpod_model_ready","local_training_state","runpod_training_state","local_training_error",
        "local_training_job_id","runpod_training_error","runpod_training_job_id","token","description"}
    if set(fields)-allowed: raise ValueError("Unsupported profile fields: "+", ".join(sorted(set(fields)-allowed)))
    if not fields: return get_face_profile_record(profile_id)
    assignments=sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in fields)
    try:
        _run(sql.SQL("UPDATE face_profiles SET {} WHERE id=%s").format(assignments), tuple(fields.values())+(profile_id,))
    except UniqueViolation as exc:
        raise ValueError("That prompt token is already assigned to another profile.") from exc
    result=get_face_profile_record(profile_id)
    if result is None: raise ValueError("Face profile not found.")
    return result


def delete_face_profile_record(profile_id):
    ensure_schema()
    return _run("DELETE FROM face_profiles WHERE id=%s",(profile_id,)) > 0


def _repair_face_photo_schema() -> None:
    """Repair created_at on legacy face_photos tables before inserting a photo."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE face_photos SET created_at = now() WHERE created_at IS NULL")
        cur.execute("ALTER TABLE face_photos ALTER COLUMN created_at SET DEFAULT now()")
        cur.execute("ALTER TABLE face_photos ALTER COLUMN created_at SET NOT NULL")


def add_face_photo(profile_id, filename, image_data, mime_type="image/jpeg", group_id=None, angle_label=None,
    photo_kind="manual", validation_status="not_checked", validation_message=None, training_caption="",
    target_yaw=None,target_pitch=None,actual_yaw=None,actual_pitch=None):
    if photo_kind not in {"source","angle","expression","body","manual"}: raise ValueError("Invalid photo kind.")
    if validation_status not in {"not_checked","passed","needs_review","not_applicable"}: raise ValueError("Invalid photo validation status.")
    ensure_schema()
    # CREATE TABLE IF NOT EXISTS does not restore defaults on an existing table.
    # Repair this required timestamp even when ensure_schema() is cached.
    _repair_face_photo_schema()
    photo_id=uuid.uuid4().hex
    # Supply the timestamp in the INSERT itself as well as repairing the legacy
    # default above. This makes the write independent of the column default.
    _run("""INSERT INTO face_photos(id,profile_id,filename,mime_type,byte_size,image_data,group_id,angle_label,photo_kind,
        validation_status,validation_message,training_caption,target_yaw,target_pitch,actual_yaw,actual_pitch,created_at)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now())""",
        (photo_id,profile_id,Path(filename).name,mime_type,len(image_data),image_data,group_id,(angle_label or "")[:80] or None,
         photo_kind,validation_status,(validation_message or "")[:1200] or None,(training_caption or "")[:1200],
         target_yaw,target_pitch,actual_yaw,actual_pitch))
    return photo_id


def _group_dict(r):
    if not r: return None
    return {"id":r["id"],"profile_id":r["profile_id"],"title":r["title"],"dataset_type":r.get("dataset_type") or "face_angles",
        "prompt":r.get("prompt") or "","status":r["status"],"expected_count":r["expected_count"],"generated_count":r["generated_count"],
        "error":r.get("error"),"created_at":_iso(r.get("created_at")),"updated_at":_iso(r.get("updated_at"))}


def _repair_face_dataset_group_schema() -> None:
    """Repair legacy required columns before creating a group, even if schema init was cached.

    CREATE TABLE IF NOT EXISTS does not repair an existing PostgreSQL table. Older
    installations may have these required columns without defaults, causing a
    dataset-job request to fail before the background worker can start.
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE face_dataset_groups SET generated_count = 0 WHERE generated_count IS NULL"
        )
        cur.execute(
            "ALTER TABLE face_dataset_groups ALTER COLUMN generated_count SET DEFAULT 0"
        )
        cur.execute(
            "ALTER TABLE face_dataset_groups ALTER COLUMN generated_count SET NOT NULL"
        )
        cur.execute(
            "UPDATE face_dataset_groups SET created_at = now() WHERE created_at IS NULL"
        )
        cur.execute(
            "ALTER TABLE face_dataset_groups ALTER COLUMN created_at SET DEFAULT now()"
        )
        cur.execute(
            "ALTER TABLE face_dataset_groups ALTER COLUMN created_at SET NOT NULL"
        )
        cur.execute(
            "UPDATE face_dataset_groups SET updated_at = now() WHERE updated_at IS NULL"
        )
        cur.execute(
            "ALTER TABLE face_dataset_groups ALTER COLUMN updated_at SET DEFAULT now()"
        )
        cur.execute(
            "ALTER TABLE face_dataset_groups ALTER COLUMN updated_at SET NOT NULL"
        )


def create_face_dataset_group(profile_id,title,expected_count=5,dataset_type="face_angles",prompt=""):
    title=(title or "").strip(); dataset_type=(dataset_type or "face_angles").strip().lower(); prompt=(prompt or "").strip()
    if not title or len(title)>120: raise ValueError("Dataset group title must contain 1–120 characters.")
    if not 1<=int(expected_count)<=30: raise ValueError("Dataset group expected count must be between 1 and 30.")
    if dataset_type not in {"face_angles","expressions","body_type"}: raise ValueError("Invalid dataset type.")
    if len(prompt)>1200: raise ValueError("Dataset prompt must be 1200 characters or fewer.")
    ensure_schema()
    _repair_face_dataset_group_schema()
    gid=uuid.uuid4().hex
    _run("INSERT INTO face_dataset_groups(id,profile_id,title,dataset_type,prompt,status,expected_count,generated_count) VALUES(%s,%s,%s,%s,%s,'generating',%s,0)",
         (gid,profile_id,title,dataset_type,prompt,int(expected_count)))
    return get_face_dataset_group(gid)


def get_face_dataset_group(group_id):
    ensure_schema(); return _group_dict(_one("SELECT * FROM face_dataset_groups WHERE id=%s",(group_id,)))


def list_face_dataset_group_photo_records(group_id):
    ensure_schema()
    return _all("""SELECT id,profile_id,filename,group_id,angle_label,photo_kind,validation_status,validation_message,training_caption,
        target_yaw,target_pitch,actual_yaw,actual_pitch FROM face_photos WHERE group_id=%s ORDER BY created_at,id""",(group_id,))


def update_face_dataset_group(group_id,*,title=None,status=None,generated_count=None,expected_count=None,error=None):
    allowed={"generating","completed","partial","failed","cancelled"}
    if status is not None and status not in allowed: raise ValueError("Invalid dataset group status.")
    fields={}
    if title is not None:
        title=str(title).strip()
        if not title or len(title)>120: raise ValueError("Dataset group name must contain 1–120 characters.")
        fields["title"]=title
    if expected_count is not None:
        expected_count=int(expected_count)
        if not 1<=expected_count<=30: raise ValueError("Expected photo count must be between 1 and 30.")
        fields["expected_count"]=expected_count
    if generated_count is not None:
        generated_count=int(generated_count)
        if generated_count<0: raise ValueError("Generated count cannot be negative.")
        fields["generated_count"]=generated_count
    if status is not None: fields["status"]=status
    if error is not None: fields["error"]=str(error)[:1200] or None
    fields["updated_at"]=_now()
    if fields:
        sets=sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in fields)
        _run(sql.SQL("UPDATE face_dataset_groups SET {} WHERE id=%s").format(sets),tuple(fields.values())+(group_id,))
    return get_face_dataset_group(group_id)


def list_face_dataset_groups(profile_id=None,limit=200):
    ensure_schema()
    if profile_id: rows=_all("SELECT g.*,p.name AS profile_name FROM face_dataset_groups g LEFT JOIN face_profiles p ON p.id=g.profile_id WHERE g.profile_id=%s ORDER BY g.created_at DESC LIMIT %s",(profile_id,max(1,min(int(limit),500))))
    else: rows=_all("SELECT g.*,p.name AS profile_name FROM face_dataset_groups g LEFT JOIN face_profiles p ON p.id=g.profile_id ORDER BY g.created_at DESC LIMIT %s",(max(1,min(int(limit),500)),))
    result=[]
    for row in rows:
        rec=_group_dict(row); rec["profile_name"]=row.get("profile_name") or "Unknown profile"
        photos=_all("""SELECT id,filename,angle_label,photo_kind,validation_status,validation_message,target_yaw,target_pitch,
            actual_yaw,actual_pitch,mime_type,byte_size,created_at FROM face_photos WHERE group_id=%s ORDER BY created_at,id""",(row["id"],))
        rec["photos"]=[dict(p,angle_label=p.get("angle_label") or "Training photo",validation_status=p.get("validation_status") or "not_checked",created_at=_iso(p.get("created_at"))) for p in photos]
        rec["photo_count"]=len(photos); result.append(rec)
    return result


def _job_dict(r):
    if not r: return None
    def parse(v):
        try:
            x=json.loads(v or "[]"); return x if isinstance(x,list) else []
        except (TypeError,json.JSONDecodeError): return []
    return {"id":r["id"],"group_id":r["group_id"],"profile_id":r["profile_id"],"dataset_type":r["dataset_type"],
        "status":r["status"],"backend":r["backend"],"expected_count":r["expected_count"],"generated_count":r["generated_count"],
        "current_index":r["current_index"],"current_label":r["current_label"],"current_task_id":r.get("current_task_id"),
        "cancel_requested":bool(r["cancel_requested"]),"group_count_base":int(r.get("group_count_base") or 0),
        "target_manifest":parse(r.get("target_manifest")),"failed_targets":parse(r.get("failed_targets")),
        "review_targets":parse(r.get("review_targets")),"source_photo_id":r["source_photo_id"],"error":r.get("error"),
        "created_at":_iso(r.get("created_at")),"updated_at":_iso(r.get("updated_at"))}


def _repair_face_dataset_job_schema() -> None:
    """Repair every defaulted NOT NULL job column on legacy PostgreSQL schemas."""
    with _connect() as conn, conn.cursor() as cur:
        # CREATE TABLE IF NOT EXISTS cannot repair columns in tables created by
        # older releases. Backfill NULLs first, then restore defaults and constraints.
        repairs = (
            ("generated_count", "0", "0"),
            ("current_index", "0", "0"),
            ("current_label", "''", "''"),
            ("group_count_base", "0", "0"),
            ("target_manifest", "'[]'", "'[]'"),
            ("cancel_requested", "FALSE", "FALSE"),
            ("failed_targets", "'[]'", "'[]'"),
            ("review_targets", "'[]'", "'[]'"),
        )
        for column, fallback, default in repairs:
            cur.execute(
                sql.SQL("UPDATE face_dataset_jobs SET {} = {} WHERE {} IS NULL").format(
                    sql.Identifier(column), sql.SQL(fallback), sql.Identifier(column)
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE face_dataset_jobs ALTER COLUMN {} SET DEFAULT {}").format(
                    sql.Identifier(column), sql.SQL(default)
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE face_dataset_jobs ALTER COLUMN {} SET NOT NULL").format(
                    sql.Identifier(column)
                )
            )
        for column in ("created_at", "updated_at"):
            cur.execute(
                sql.SQL("UPDATE face_dataset_jobs SET {} = now() WHERE {} IS NULL").format(
                    sql.Identifier(column), sql.Identifier(column)
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE face_dataset_jobs ALTER COLUMN {} SET DEFAULT now()").format(
                    sql.Identifier(column)
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE face_dataset_jobs ALTER COLUMN {} SET NOT NULL").format(
                    sql.Identifier(column)
                )
            )


def create_face_dataset_job(*,job_id,group_id,profile_id,dataset_type,backend,expected_count,source_photo_id,target_manifest=None,group_count_base=0):
    if dataset_type not in {"face_angles","expressions"}: raise ValueError("Unsupported face dataset generation type.")
    if backend not in {"cpu","local_gpu","runpod","cloud_gpu"}: raise ValueError("Unsupported generation backend.")
    if not 1<=int(expected_count)<=20: raise ValueError("Expected target count must be between 1 and 20.")
    targets=[x for x in (target_manifest or []) if isinstance(x,dict)]
    if targets and len(targets)!=int(expected_count): raise ValueError("Target manifest count must match expected_count.")
    if int(group_count_base)<0: raise ValueError("Base group photo count cannot be negative.")
    if not all(isinstance(x,str) and len(x)==32 for x in (job_id,group_id,profile_id,source_photo_id)): raise ValueError("Invalid dataset generation job identifiers.")
    ensure_schema()
    _repair_face_dataset_job_schema()
    group=get_face_dataset_group(group_id)
    if not group or group["profile_id"]!=profile_id or group["dataset_type"]!=dataset_type: raise ValueError("Dataset group does not match generation job.")
    source=_one("SELECT id FROM face_photos WHERE id=%s AND profile_id=%s AND group_id=%s",(source_photo_id,profile_id,group_id))
    if not source: raise ValueError("Source photo does not belong to target dataset group.")
    _run("""INSERT INTO face_dataset_jobs(
        id,group_id,profile_id,dataset_type,status,backend,expected_count,generated_count,
        current_index,current_label,source_photo_id,group_count_base,target_manifest,
        cancel_requested,failed_targets,review_targets,created_at,updated_at)
        VALUES(%s,%s,%s,%s,'queued',%s,%s,0,0,'',%s,%s,%s,FALSE,'[]','[]',now(),now())""",
        (job_id,group_id,profile_id,dataset_type,backend,int(expected_count),source_photo_id,
         int(group_count_base),json.dumps(targets,ensure_ascii=False)))
    return get_face_dataset_job(job_id)


def get_face_dataset_job(job_id):
    ensure_schema(); return _job_dict(_one("SELECT * FROM face_dataset_jobs WHERE id=%s",(job_id,)))


def list_face_dataset_jobs(*,profile_id=None,group_id=None,active_only=False,limit=50):
    ensure_schema(); clauses=[]; vals=[]
    if profile_id: clauses.append("profile_id=%s"); vals.append(profile_id)
    if group_id: clauses.append("group_id=%s"); vals.append(group_id)
    if active_only: clauses.append("status IN ('queued','running','cancelling')")
    where=(" WHERE "+" AND ".join(clauses)) if clauses else ""
    vals.append(max(1,min(int(limit),100)))
    return [_job_dict(r) for r in _all("SELECT * FROM face_dataset_jobs"+where+" ORDER BY created_at DESC LIMIT %s",tuple(vals))]


def update_face_dataset_job(job_id,**fields):
    allowed={"status","generated_count","current_index","current_label","current_task_id","cancel_requested","failed_targets","review_targets","error"}
    if set(fields)-allowed: raise ValueError("Unsupported job fields: "+", ".join(sorted(set(fields)-allowed)))
    if "status" in fields and fields["status"] not in {"queued","running","cancelling","cancelled","completed","partial","failed","interrupted"}: raise ValueError("Invalid training job state.")
    for k in ("generated_count","current_index"):
        if k in fields and int(fields[k])<0: raise ValueError(k+" cannot be negative.")
    vals={}
    for k,v in fields.items():
        if k in {"failed_targets","review_targets"}: v=json.dumps(v if isinstance(v,list) else [],ensure_ascii=False)
        elif k=="error": v=str(v or "")[:2000] or None
        elif k=="current_label": v=str(v or "")[:80]
        vals[k]=v
    if vals:
        vals["updated_at"]=_now()
        sets=sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in vals)
        _run(sql.SQL("UPDATE face_dataset_jobs SET {} WHERE id=%s").format(sets),tuple(vals.values())+(job_id,))
    return get_face_dataset_job(job_id)


def update_face_photo_validation(photo_id,*,validation_status,validation_message="",actual_yaw=None,actual_pitch=None):
    if validation_status not in {"not_checked","passed","needs_review","not_applicable"}: raise ValueError("Invalid photo validation status.")
    _run("UPDATE face_photos SET validation_status=%s,validation_message=%s,actual_yaw=%s,actual_pitch=%s WHERE id=%s",
         (validation_status,(validation_message or "")[:1200] or None,actual_yaw,actual_pitch,photo_id))
    return _one("SELECT id,validation_status,validation_message,actual_yaw,actual_pitch FROM face_photos WHERE id=%s",(photo_id,))


def delete_face_dataset_group(group_id):
    ensure_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM face_dataset_groups WHERE id=%s",(group_id,)); group=cur.fetchone()
        if not group: return None
        cur.execute("SELECT id,profile_id,filename FROM face_photos WHERE group_id=%s",(group_id,)); photos=cur.fetchall()
        cur.execute("DELETE FROM face_photos WHERE group_id=%s",(group_id,))
        cur.execute("DELETE FROM face_dataset_groups WHERE id=%s",(group_id,))
    result=_group_dict(group); result["photos"]=photos; return result


def list_face_photo_names(profile_id):
    ensure_schema(); return {r["filename"] for r in _all("SELECT filename FROM face_photos WHERE profile_id=%s",(profile_id,))}


def _photo_dict(r,include_bytes=False):
    if not r:return None
    keys=("id","profile_id","filename","mime_type","byte_size","group_id","angle_label","photo_kind","validation_status",
          "validation_message","training_caption","target_yaw","target_pitch","actual_yaw","actual_pitch")
    d={k:r.get(k) for k in keys}
    d["validation_status"]=d.get("validation_status") or "not_checked"; d["training_caption"]=d.get("training_caption") or ""
    if include_bytes:d["image_data"]=bytes(r["image_data"])
    return d


def list_face_photo_records(profile_id,filenames=None):
    ensure_schema()
    if filenames is not None:
        if not filenames:return []
        rows=_all("SELECT * FROM face_photos WHERE profile_id=%s AND filename=ANY(%s) ORDER BY created_at",(profile_id,list(filenames)))
    else: rows=_all("SELECT * FROM face_photos WHERE profile_id=%s ORDER BY created_at",(profile_id,))
    return [_photo_dict(r,True) for r in rows]


def get_face_photo_bytes(photo_id):
    ensure_schema(); r=_one("SELECT * FROM face_photos WHERE id=%s",(photo_id,))
    return (_photo_dict(r),bytes(r["image_data"])) if r else None


def update_face_photo(photo_id,*,angle_label=None,training_caption=None,image_data=None,mime_type=None):
    if image_data is not None and (not image_data or len(image_data)>8*1024*1024): raise ValueError("Replacement photos must be non-empty and 8 MB or smaller.")
    if angle_label is not None:
        angle_label=str(angle_label).strip()
        if not angle_label or len(angle_label)>80: raise ValueError("Photo label must contain 1–80 characters.")
    updates={}
    if angle_label is not None: updates["angle_label"]=angle_label
    if training_caption is not None: updates["training_caption"]=str(training_caption or "").strip()[:1200]
    if image_data is not None:
        updates.update(image_data=image_data,byte_size=len(image_data),mime_type=(mime_type or "image/jpeg")[:100],
            validation_status="needs_review",validation_message="The image was replaced; review its identity, pose and expression before training.",
            actual_yaw=None,actual_pitch=None)
    if updates:
        sets=sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in updates)
        _run(sql.SQL("UPDATE face_photos SET {} WHERE id=%s").format(sets),tuple(updates.values())+(photo_id,))
    r=_one("SELECT * FROM face_photos WHERE id=%s",(photo_id,))
    if not r:return None
    d=_photo_dict(r); d["angle_label"]=d.get("angle_label") or "Training photo"; return d


def delete_face_photo(photo_id):
    ensure_schema(); r=_one("SELECT id,profile_id,filename,group_id,angle_label,photo_kind,validation_status FROM face_photos WHERE id=%s",(photo_id,))
    if r:_run("DELETE FROM face_photos WHERE id=%s",(photo_id,))
    return r


def save_face_model_bundle(profile_id,backend,filename,bundle_data,file_count):
    if backend not in {"local","runpod"}: raise ValueError("Face model backend must be local or runpod.")
    if not bundle_data or len(bundle_data)>512*1024*1024: raise ValueError("Face model bundle must be non-empty and no larger than 512 MB.")
    ensure_schema(); name=Path(filename).name
    r=_one("""INSERT INTO face_model_artifacts(id,profile_id,backend,filename,file_count,bundle_size,bundle_data)
        VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(profile_id,backend) DO UPDATE SET filename=EXCLUDED.filename,
        file_count=EXCLUDED.file_count,bundle_size=EXCLUDED.bundle_size,bundle_data=EXCLUDED.bundle_data,updated_at=now()
        RETURNING id,profile_id,backend,filename,file_count,bundle_size,updated_at""",
        (uuid.uuid4().hex,profile_id,backend,name,int(file_count),len(bundle_data),bundle_data))
    return dict(r,updated_at=_iso(r["updated_at"]))


def get_face_model_bundle(profile_id,backend):
    ensure_schema(); r=_one("SELECT * FROM face_model_artifacts WHERE profile_id=%s AND backend=%s",(profile_id,backend))
    if not r:return None
    r=dict(r,updated_at=_iso(r["updated_at"]),bundle_data=bytes(r["bundle_data"])); return r


def _asset_dict(r,include_content=False):
    if not r:return None
    d={k:r.get(k) for k in ("id","asset_type","title","filename","mime_type","prompt","negative_prompt","profile_id","profile_name","backend","generation_mode","media_size")}
    d.update(created_at=_iso(r.get("created_at")),updated_at=_iso(r.get("updated_at")))
    if include_content:d["media_data"]=bytes(r["media_data"])
    return d


def create_asset(*,asset_type,title,filename,mime_type,media_data,prompt="",negative_prompt="",profile_id=None,profile_name=None,backend="cpu",generation_mode=""):
    if asset_type not in {"image","video"}: raise ValueError("Asset type must be image or video.")
    if not media_data: raise ValueError("Cannot save an empty media file.")
    if len(media_data)>512*1024*1024: raise ValueError("Media files must be 512 MB or smaller to store in the database.")
    ensure_schema()
    title=(title or "").strip()[:160] or ("Generated image" if asset_type=="image" else "Generated video")
    # Explicitly set timestamps instead of relying on defaults that may be
    # absent on an existing Supabase table created by an older schema version.
    r=_one("""INSERT INTO generated_assets(id,asset_type,title,filename,mime_type,prompt,negative_prompt,profile_id,profile_name,
        backend,generation_mode,media_size,media_data,created_at,updated_at)
        VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now()) RETURNING *""",
        (uuid.uuid4().hex,asset_type,title,Path(filename).name,mime_type,prompt[:1200],negative_prompt[:600],profile_id,profile_name,
         backend[:32],generation_mode[:32],len(media_data),media_data))
    return _asset_dict(r)


def list_asset_filenames(asset_type):
    if asset_type not in {"image","video"}: raise ValueError("Asset type must be image or video.")
    ensure_schema(); return {Path(r["filename"]).name for r in _all("SELECT filename FROM generated_assets WHERE asset_type=%s",(asset_type,))}


def _asset_where(asset_type=None,search=""):
    clauses=[]; vals=[]
    if asset_type in {"image","video"}: clauses.append("asset_type=%s"); vals.append(asset_type)
    term=(search or "").strip()[:160]
    if term:
        clauses.append("(title ILIKE %s OR prompt ILIKE %s OR negative_prompt ILIKE %s OR COALESCE(profile_name,'') ILIKE %s)")
        pattern="%"+term+"%"; vals.extend([pattern]*4)
    return (" WHERE "+" AND ".join(clauses) if clauses else ""),tuple(vals)


def count_assets(asset_type=None,search=""):
    ensure_schema(); where,vals=_asset_where(asset_type,search)
    return int(_one("SELECT count(*) AS n FROM generated_assets"+where,vals)["n"])


def list_assets(asset_type=None,limit=200,offset=0,search="",sort="newest"):
    ensure_schema(); where,vals=_asset_where(asset_type,search)
    order="created_at ASC,id ASC" if sort=="oldest" else ("lower(title) ASC,created_at DESC" if sort=="title" else "created_at DESC,id DESC")
    rows=_all("SELECT * FROM generated_assets"+where+" ORDER BY "+order+" LIMIT %s OFFSET %s",vals+(max(1,min(int(limit),500)),max(0,int(offset))))
    return [_asset_dict(r) for r in rows]


def get_asset(asset_id,include_content=False):
    ensure_schema(); return _asset_dict(_one("SELECT * FROM generated_assets WHERE id=%s",(asset_id,)),include_content)


def update_asset(asset_id,fields):
    allowed={"title","prompt","negative_prompt"}
    if not fields or set(fields)-allowed: raise ValueError("Update title, prompt, and/or negative prompt only.")
    fields=dict(fields)
    if "title" in fields:
        fields["title"]=str(fields["title"] or "").strip()
        if not fields["title"] or len(fields["title"])>160: raise ValueError("Title must contain 1–160 characters.")
    if "prompt" in fields and len(str(fields["prompt"] or ""))>1200: raise ValueError("Prompt must be 1200 characters or fewer.")
    if "negative_prompt" in fields and len(str(fields["negative_prompt"] or ""))>600: raise ValueError("Negative prompt must be 600 characters or fewer.")
    sets=sql.SQL(", ").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in fields)
    r=_one(sql.SQL("UPDATE generated_assets SET {} , updated_at=now() WHERE id=%s RETURNING *").format(sets),tuple(fields.values())+(asset_id,))
    return _asset_dict(r)


def delete_asset(asset_id):
    ensure_schema(); r=_one("SELECT * FROM generated_assets WHERE id=%s",(asset_id,))
    if r:_run("DELETE FROM generated_assets WHERE id=%s",(asset_id,))
    return _asset_dict(r)
