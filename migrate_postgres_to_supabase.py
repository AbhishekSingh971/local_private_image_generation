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
        # Create/upgrade the known application schema in the target through the
        # application's database layer. This runs only against DATABASE_URL.
        os.environ["DATABASE_URL"] = target_uri
        from studio_db import ensure_schema
        ensure_schema()

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


if __name__ == "__main__":
    raise SystemExit(main())
