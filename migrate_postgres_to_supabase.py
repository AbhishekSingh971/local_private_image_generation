"""Copy the application's PostgreSQL data into the Supabase target database.

Configure OLD_DATABASE_URL to the source PostgreSQL database and DATABASE_URL to
the Supabase PostgreSQL database. This script never deletes or overwrites target
rows: conflicts on primary keys or unique constraints are skipped and reported.

Run from the project root:
    python migrate_postgres_to_supabase.py
"""
from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
from psycopg.rows import tuple_row

# Parent-before-child order preserves foreign-key relationships.
TABLES = (
    "face_profiles",
    "face_dataset_groups",
    "face_photos",
    "face_dataset_jobs",
    "face_model_artifacts",
    "generated_assets",
)


def _connection_uri(value: str, label: str) -> str:
    value = (value or "").strip()
    if not value:
        raise RuntimeError(f"{label} is missing. Set it in your private environment.")
    if value.startswith("postgres://"):
        value = "postgresql://" + value[len("postgres://"):]
    if value.startswith("postgresql+psycopg://"):
        value = "postgresql://" + value[len("postgresql+psycopg://"):]
    if not value.startswith(("postgresql://", "postgres://")):
        raise RuntimeError(f"{label} must be a PostgreSQL connection URI.")
    return value


def _table_exists(conn, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name = %s)",
            (table,),
        )
        return bool(cur.fetchone()[0])


def _columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = %s "
            "ORDER BY ordinal_position",
            (table,),
        )
        return [row[0] for row in cur.fetchall()]


def _row_count(conn, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
        return int(cur.fetchone()[0])


def migrate(source_uri: str, target_uri: str) -> int:
    # Guard against accidentally treating the same database as source and target.
    source_parts = urlsplit(source_uri)
    target_parts = urlsplit(target_uri)
    source_identity = (source_parts.hostname, source_parts.port, source_parts.path, source_parts.username)
    target_identity = (target_parts.hostname, target_parts.port, target_parts.path, target_parts.username)
    if source_identity == target_identity:
        raise RuntimeError("Source and target appear to be the same database; migration aborted.")

    with psycopg.connect(source_uri, row_factory=tuple_row) as source, psycopg.connect(
        target_uri, row_factory=tuple_row
    ) as target:
        # Create the target schema directly with psycopg. The migration utility
        # does not depend on SQLAlchemy or import the application's runtime.
        _ensure_target_schema(target)

        total_inserted = 0
        total_skipped = 0
        print("PostgreSQL to Supabase migration (non-destructive)")
        print("Target rows are never updated or deleted. Duplicate/conflicting rows are skipped.")
        for table in TABLES:
            if not _table_exists(source, table):
                print(f"- {table}: absent in source; skipped")
                continue
            if not _table_exists(target, table):
                print(f"- {table}: absent in Supabase after schema setup; skipped")
                continue

            source_columns = _columns(source, table)
            target_columns = set(_columns(target, table))
            columns = [name for name in source_columns if name in target_columns]
            if not columns:
                print(f"- {table}: no matching columns; skipped")
                continue

            before = _row_count(target, table)
            inserted = 0
            skipped = 0
            column_sql = sql.SQL(", ").join(sql.Identifier(name) for name in columns)
            placeholders = sql.SQL(", ").join(sql.Placeholder() for _ in columns)
            statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
                sql.Identifier(table), column_sql, placeholders
            )

            with source.cursor(name=f"migration_{table}") as src_cur:
                src_cur.itersize = 250
                src_cur.execute(
                    sql.SQL("SELECT {} FROM {}").format(
                        column_sql, sql.Identifier(table)
                    )
                )
                with target.cursor() as dst_cur:
                    for row in src_cur:
                        dst_cur.execute(statement, row)
                        if dst_cur.rowcount == 1:
                            inserted += 1
                        else:
                            skipped += 1
                target.commit()

            after = _row_count(target, table)
            # The delta is a useful safety check, including when concurrent writes occur.
            print(
                f"- {table}: source rows={_row_count(source, table)}, "
                f"inserted={inserted}, skipped/conflicts={skipped}, "
                f"target rows before={before}, after={after}"
            )
            total_inserted += inserted
            total_skipped += skipped

        print(f"Finished. Inserted={total_inserted}; skipped/conflicts={total_skipped}.")
        print("Review the counts above before changing or removing the old database.")
    return 0


def main() -> int:
    try:
        source = _connection_uri(os.environ.get("OLD_DATABASE_URL", ""), "OLD_DATABASE_URL")
        target = _connection_uri(os.environ.get("DATABASE_URL", ""), "DATABASE_URL")
        if source == target:
            raise RuntimeError("OLD_DATABASE_URL and DATABASE_URL must be different.")
        return migrate(source, target)
    except Exception as exc:
        # Do not print connection URIs or exception strings that may contain secrets.
        print(f"Migration failed: {type(exc).__name__}. Check connectivity, credentials, and schema.", file=sys.stderr)
        return 1


def _ensure_target_schema(conn) -> None:
    """Create the application's PostgreSQL tables if absent; preserve existing data."""
    statements = (
        """CREATE TABLE IF NOT EXISTS face_profiles (
            id VARCHAR(32) PRIMARY KEY, name VARCHAR(80) NOT NULL,
            name_key VARCHAR(80) UNIQUE NOT NULL, token VARCHAR(48) UNIQUE NOT NULL,
            description TEXT NOT NULL DEFAULT '', created_at VARCHAR(40) NOT NULL,
            local_model_ready BOOLEAN NOT NULL DEFAULT FALSE,
            runpod_model_ready BOOLEAN NOT NULL DEFAULT FALSE,
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
            actual_yaw DOUBLE PRECISION, actual_pitch DOUBLE PRECISION,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now())""",
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
        "CREATE INDEX IF NOT EXISTS ix_face_dataset_groups_profile_id ON face_dataset_groups(profile_id)",
        "CREATE INDEX IF NOT EXISTS ix_face_photos_profile_id ON face_photos(profile_id)",
        "CREATE INDEX IF NOT EXISTS ix_face_photos_group_id ON face_photos(group_id)",
        "CREATE INDEX IF NOT EXISTS ix_face_dataset_jobs_group_id ON face_dataset_jobs(group_id)",
        "CREATE INDEX IF NOT EXISTS ix_face_dataset_jobs_profile_id ON face_dataset_jobs(profile_id)",
        "CREATE INDEX IF NOT EXISTS ix_face_dataset_jobs_status ON face_dataset_jobs(status)",
        "CREATE INDEX IF NOT EXISTS ix_face_model_artifacts_profile_id ON face_model_artifacts(profile_id)",
        "CREATE INDEX IF NOT EXISTS ix_generated_assets_asset_type ON generated_assets(asset_type)",
        "CREATE INDEX IF NOT EXISTS ix_generated_assets_profile_id ON generated_assets(profile_id)",
    )
    with conn.cursor() as cur:
        # Defer indexes until compatibility columns have been added to older tables.
        index_statements = [statement for statement in statements if statement.startswith("CREATE INDEX")]
        for statement in statements:
            if not statement.startswith("CREATE INDEX"):
                cur.execute(statement)
        # Add columns introduced by newer app versions to older target tables.
        cur.execute("ALTER TABLE face_profiles ADD COLUMN IF NOT EXISTS description TEXT NOT NULL DEFAULT ''")
        cur.execute("ALTER TABLE face_dataset_groups ADD COLUMN IF NOT EXISTS dataset_type VARCHAR(24) NOT NULL DEFAULT 'face_angles'")
        cur.execute("ALTER TABLE face_dataset_groups ADD COLUMN IF NOT EXISTS prompt TEXT NOT NULL DEFAULT ''")
        cur.execute("ALTER TABLE face_dataset_jobs ADD COLUMN IF NOT EXISTS group_count_base INTEGER NOT NULL DEFAULT 0")
        cur.execute("ALTER TABLE face_dataset_jobs ADD COLUMN IF NOT EXISTS target_manifest TEXT NOT NULL DEFAULT '[]'")
        for name, definition in (
            ("group_id", "VARCHAR(32)"), ("angle_label", "VARCHAR(80)"),
            ("photo_kind", "VARCHAR(16) NOT NULL DEFAULT 'manual'"),
            ("validation_status", "VARCHAR(24) NOT NULL DEFAULT 'not_checked'"),
            ("validation_message", "TEXT"), ("training_caption", "TEXT NOT NULL DEFAULT ''"),
            ("target_yaw", "DOUBLE PRECISION"), ("target_pitch", "DOUBLE PRECISION"),
            ("actual_yaw", "DOUBLE PRECISION"), ("actual_pitch", "DOUBLE PRECISION"),
        ):
            cur.execute(sql.SQL("ALTER TABLE face_photos ADD COLUMN IF NOT EXISTS {} ").format(sql.Identifier(name)) + sql.SQL(definition))
        for statement in index_statements:
            cur.execute(statement)
    conn.commit()


if __name__ == "__main__":
    raise SystemExit(main())

