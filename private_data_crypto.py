"""Application-layer encryption for private face photos, LoRAs and media blobs."""
from __future__ import annotations

import base64
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"PFDB1"
NONCE_SIZE = 12
AAD = b"PrivateFaceStudio:private-data:v1"


def _decode_key(value: str | None) -> bytes | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        key = bytes.fromhex(raw) if len(raw) == 64 else base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    except (ValueError, TypeError) as exc:
        raise RuntimeError("Private-data encryption keys must be 64 hex characters or URL-safe base64 for 32 bytes.") from exc
    if len(key) != 32:
        raise RuntimeError("Private-data encryption keys must decode to exactly 32 bytes.")
    return key


def active_encryption_key() -> bytes | None:
    return _decode_key(os.environ.get("FACE_DATA_ENCRYPTION_KEY"))


def encryption_configured() -> bool:
    return active_encryption_key() is not None


def is_encrypted(payload: bytes) -> bool:
    return bytes(payload).startswith(MAGIC)


def encrypt_sensitive_bytes(payload: bytes) -> bytes:
    raw = bytes(payload)
    key = active_encryption_key()
    if key is None or is_encrypted(raw):
        return raw
    nonce = secrets.token_bytes(NONCE_SIZE)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, raw, AAD)


def decrypt_sensitive_bytes(payload: bytes) -> bytes:
    raw = bytes(payload)
    if not is_encrypted(raw):
        return raw
    nonce_start = len(MAGIC)
    nonce = raw[nonce_start:nonce_start + NONCE_SIZE]
    ciphertext = raw[nonce_start + NONCE_SIZE:]
    if len(nonce) != NONCE_SIZE or len(ciphertext) < 16:
        raise RuntimeError("Encrypted private data is truncated or corrupt.")
    keys = [
        _decode_key(os.environ.get("FACE_DATA_ENCRYPTION_KEY")),
        _decode_key(os.environ.get("FACE_DATA_ENCRYPTION_KEY_PREVIOUS")),
    ]
    for key in keys:
        if key is None:
            continue
        try:
            return AESGCM(key).decrypt(nonce, ciphertext, AAD)
        except Exception:
            continue
    raise RuntimeError(
        "Could not decrypt private data. Configure the original FACE_DATA_ENCRYPTION_KEY "
        "or add it temporarily as FACE_DATA_ENCRYPTION_KEY_PREVIOUS."
    )
