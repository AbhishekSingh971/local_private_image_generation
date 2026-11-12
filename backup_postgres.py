"""Encrypted PostgreSQL backup and restore utility for private studio data.

Requires pg_dump/pg_restore on PATH and BACKUP_ENCRYPTION_KEY in the environment.
The backup includes PostgreSQL rows and bytea media/model bundles. It does not
back up base model checkpoints, which should be managed as reproducible downloads.
"""
from __future__ import annotations

import argparse
import base64
import os
import secrets
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit, parse_qs

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

MAGIC = b"PFSA1"
NONCE_SIZE = 12
TAG_SIZE = 16
CHUNK_SIZE = 1024 * 1024


def load_encryption_key(value: str | None = None) -> bytes:
    raw = (value or os.environ.get("BACKUP_ENCRYPTION_KEY") or "").strip()
    if not raw:
        raise RuntimeError("Set BACKUP_ENCRYPTION_KEY to a 32-byte hex or URL-safe base64 key.")
    try:
        if len(raw) == 64:
            key = bytes.fromhex(raw)
        else:
            key = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("BACKUP_ENCRYPTION_KEY must be 64 hex characters or URL-safe base64 for 32 bytes.") from exc
    if len(key) != 32:
        raise RuntimeError("BACKUP_ENCRYPTION_KEY must decode to exactly 32 bytes.")
    return key


def encrypt_bytes(payload: bytes, key: bytes, nonce: bytes | None = None) -> bytes:
    if len(key) != 32:
        raise ValueError("AES-256 backup key must be exactly 32 bytes.")
    iv = nonce or secrets.token_bytes(NONCE_SIZE)
    encryptor = Cipher(algorithms.AES(key), modes.GCM(iv)).encryptor()
    ciphertext = encryptor.update(payload) + encryptor.finalize()
    return MAGIC + iv + ciphertext + encryptor.tag


def decrypt_bytes(payload: bytes, key: bytes) -> bytes:
    if len(key) != 32 or len(payload) < len(MAGIC) + NONCE_SIZE + TAG_SIZE or not payload.startswith(MAGIC):
        raise ValueError("Backup file header or encryption key is invalid.")
    iv_start = len(MAGIC)
    nonce = payload[iv_start:iv_start + NONCE_SIZE]
    ciphertext = payload[iv_start + NONCE_SIZE:-TAG_SIZE]
    tag = payload[-TAG_SIZE:]
    decryptor = Cipher(algorithms.AES(key), modes.GCM(nonce, tag)).decryptor()
    try:
        return decryptor.update(ciphertext) + decryptor.finalize()
    except Exception as exc:
        raise ValueError("Backup authentication failed. Check the encryption key and file integrity.") from exc


def _postgres_environment(database_url: str) -> tuple[list[str], dict[str, str]]:
    parsed = urlsplit(database_url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise ValueError("DATABASE_URL must be a PostgreSQL URI.")
    database = unquote(parsed.path.lstrip("/"))
    if not database:
        raise ValueError("DATABASE_URL does not include a database name.")
    args = [
        "--host", parsed.hostname,
        "--port", str(parsed.port or 5432),
        "--username", unquote(parsed.username or ""),
        "--no-password",
    ]
    env = os.environ.copy()
    if parsed.password is not None:
        env["PGPASSWORD"] = unquote(parsed.password)
    query = parse_qs(parsed.query)
    if query.get("sslmode"):
        env["PGSSLMODE"] = query["sslmode"][0]
    return args + ["--dbname", database], env


def create_backup(output_path: str | Path, *, database_url: str | None = None, key: bytes | None = None) -> Path:
    url = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("Set DATABASE_URL before creating a database backup.")
    encryption_key = key or load_encryption_key()
    args, env = _postgres_environment(url)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_bytes(NONCE_SIZE)
    encryptor = Cipher(algorithms.AES(encryption_key), modes.GCM(nonce)).encryptor()
    temporary = destination.with_name(destination.name + ".tmp")
    command = ["pg_dump", *args, "--format=custom", "--no-owner", "--no-acl"]
    try:
        with temporary.open("wb") as target:
            target.write(MAGIC + nonce)
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
            assert process.stdout is not None
            while True:
                chunk = process.stdout.read(CHUNK_SIZE)
                if not chunk:
                    break
                target.write(encryptor.update(chunk))
            target.write(encryptor.finalize())
            stderr = process.stderr.read() if process.stderr else b""
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError("pg_dump failed: " + stderr.decode("utf-8", "replace")[-1200:])
            target.write(encryptor.tag)
        temporary.replace(destination)
        return destination
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def restore_backup(backup_path: str | Path, *, database_url: str | None = None, key: bytes | None = None) -> None:
    url = (database_url or os.environ.get("DATABASE_URL") or "").strip()
    if not url:
        raise RuntimeError("Set DATABASE_URL before restoring a database backup.")
    encryption_key = key or load_encryption_key()
    source = Path(backup_path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(source)
    args, env = _postgres_environment(url)
    with tempfile.TemporaryDirectory(prefix="private-studio-restore-") as temp:
        decrypted = Path(temp) / "database.dump"
        raw_size = source.stat().st_size
        if raw_size < len(MAGIC) + NONCE_SIZE + TAG_SIZE:
            raise ValueError("Backup file is too small or corrupt.")
        with source.open("rb") as src, decrypted.open("wb") as dst:
            header = src.read(len(MAGIC) + NONCE_SIZE)
            if not header.startswith(MAGIC):
                raise ValueError("Backup file format is not recognized.")
            nonce = header[len(MAGIC):]
            src.seek(-TAG_SIZE, os.SEEK_END)
            tag = src.read(TAG_SIZE)
            src.seek(len(MAGIC) + NONCE_SIZE)
            remaining = raw_size - len(MAGIC) - NONCE_SIZE - TAG_SIZE
            decryptor = Cipher(algorithms.AES(encryption_key), modes.GCM(nonce, tag)).decryptor()
            while remaining:
                chunk = src.read(min(CHUNK_SIZE, remaining))
                if not chunk:
                    raise ValueError("Backup file ended unexpectedly.")
                remaining -= len(chunk)
                dst.write(decryptor.update(chunk))
            try:
                dst.write(decryptor.finalize())
            except Exception as exc:
                raise ValueError("Backup authentication failed. Check the encryption key and file integrity.") from exc
        command = ["pg_restore", *args, "--clean", "--if-exists", "--no-owner", "--no-acl", str(decrypted)]
        completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=3600, check=False)
        if completed.returncode != 0:
            raise RuntimeError("pg_restore failed: " + (completed.stderr or "")[-1600:])


def main():
    parser = argparse.ArgumentParser(description="Create or restore an encrypted private-studio PostgreSQL backup.")
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--output", type=Path, default=Path("backups") / "private-studio.pfbackup")
    restore = sub.add_parser("restore")
    restore.add_argument("backup", type=Path)
    restore.add_argument("--confirm-restore", action="store_true", help="Required because restore replaces database objects.")
    args = parser.parse_args()
    if args.action == "create":
        path = create_backup(args.output)
        print(f"Encrypted PostgreSQL backup created: {path} ({path.stat().st_size} bytes)")
    else:
        if not args.confirm_restore:
            raise SystemExit("Restore is destructive. Re-run with --confirm-restore after checking the target DATABASE_URL.")
        restore_backup(args.backup)
        print("PostgreSQL backup restored successfully.")


if __name__ == "__main__":
    main()
