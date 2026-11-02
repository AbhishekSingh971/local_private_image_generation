"""PostgreSQL persistence for reward observations and contextual bandit state."""
from __future__ import annotations
import json, os, uuid
from datetime import timezone
from typing import Any, Mapping
import psycopg
from psycopg.rows import dict_row

def _database_url() -> str:
    value = (os.getenv("DATABASE_URL") or "").strip()
    if value.startswith("postgres://"): value = "postgresql://" + value[len("postgres://"):]
    if value.startswith("postgresql+psycopg://"): value = "postgresql://" + value[len("postgresql+psycopg://"):]
    if not value.startswith("postgresql://"):
        raise RuntimeError("DATABASE_URL must be configured with a PostgreSQL URI.")
    return value

def _connect():
    return psycopg.connect(_database_url(), row_factory=dict_row, connect_timeout=10)

def ensure_feedback_schema() -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""CREATE TABLE IF NOT EXISTS generation_feedback (
            id VARCHAR(32) PRIMARY KEY, asset_id VARCHAR(32), profile_id VARCHAR(32),
            task VARCHAR(32) NOT NULL, configuration_id VARCHAR(120) NOT NULL,
            context JSONB NOT NULL DEFAULT '{}'::jsonb, scores JSONB NOT NULL DEFAULT '{}'::jsonb,
            reward DOUBLE PRECISION NOT NULL CHECK (reward >= -1 AND reward <= 1),
            sources JSONB NOT NULL DEFAULT '{}'::jsonb, penalties JSONB NOT NULL DEFAULT '[]'::jsonb,
            needs_human_review JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())""")
        cur.execute("ALTER TABLE generation_feedback ADD COLUMN IF NOT EXISTS source_id VARCHAR(32)")
        cur.execute("ALTER TABLE generation_feedback ADD COLUMN IF NOT EXISTS source_type VARCHAR(16)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_generation_feedback_asset ON generation_feedback(asset_id, created_at DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS ix_generation_feedback_source ON generation_feedback(source_id, created_at DESC)")
        cur.execute("""CREATE TABLE IF NOT EXISTS generation_bandit_state (
            context_key VARCHAR(200) PRIMARY KEY, state JSONB NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now())""")

def save_feedback(*, asset_id: str | None, profile_id: str | None, source_id: str | None, source_type: str | None, task: str,
                  configuration_id: str, context: Mapping[str, Any],
                  report: Mapping[str, Any]) -> str:
    ensure_feedback_schema()
    feedback_id = uuid.uuid4().hex
    reward = float(report["reward"])
    if not -1.0 <= reward <= 1.0: raise ValueError("Reward must be normalized to [-1, 1].")
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO generation_feedback(
            id,asset_id,profile_id,source_id,source_type,task,configuration_id,context,scores,reward,sources,penalties,needs_human_review)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s,%s::jsonb,%s::jsonb,%s::jsonb)""",
            (feedback_id, asset_id, profile_id, source_id, source_type, task[:32], configuration_id[:120],
             json.dumps(dict(context)), json.dumps(dict(report.get("scores", {}))), reward,
             json.dumps(dict(report.get("sources", {}))), json.dumps(list(report.get("penalties", []))),
             json.dumps(list(report.get("needs_human_review", [])))))
    return feedback_id

def load_bandit_state(context_key: str) -> dict[str, Any] | None:
    ensure_feedback_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT state FROM generation_bandit_state WHERE context_key=%s", (context_key[:200],))
        row = cur.fetchone()
    return dict(row["state"]) if row else None

def save_bandit_state(context_key: str, state: Mapping[str, Any]) -> None:
    ensure_feedback_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO generation_bandit_state(context_key,state,updated_at)
            VALUES (%s,%s::jsonb,now()) ON CONFLICT(context_key) DO UPDATE
            SET state=EXCLUDED.state,updated_at=now()""",
            (context_key[:200], json.dumps(dict(state))))

def list_feedback(asset_id: str, limit: int = 50) -> list[dict[str, Any]]:
    ensure_feedback_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT id,asset_id,profile_id,task,configuration_id,context,scores,reward,
            sources,penalties,needs_human_review,created_at FROM generation_feedback
            WHERE asset_id=%s ORDER BY created_at DESC LIMIT %s""",
            (asset_id, max(1, min(int(limit), 100))))
        rows = cur.fetchall()
    return [{**dict(row), "created_at": row["created_at"].astimezone(timezone.utc).isoformat()} for row in rows]

def list_feedback_for_source(source_id: str, limit: int = 30) -> list[dict[str, Any]]:
    ensure_feedback_schema()
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT id,source_id,source_type,profile_id,task,configuration_id,
            scores,reward,sources,penalties,needs_human_review,created_at
            FROM generation_feedback WHERE source_id=%s
            ORDER BY created_at DESC LIMIT %s""",
            (source_id, max(1, min(int(limit), 100))))
        rows = cur.fetchall()
    return [{**dict(row), "created_at": row["created_at"].astimezone(timezone.utc).isoformat()} for row in rows]
