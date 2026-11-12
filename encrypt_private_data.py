"""Encrypt legacy plaintext media and face-model BYTEA blobs in PostgreSQL."""
from __future__ import annotations

import argparse
import json

from private_data_crypto import encryption_configured
from studio_db import encrypt_existing_private_data, private_data_encryption_status


def main():
    parser = argparse.ArgumentParser(description="Encrypt existing private studio data in PostgreSQL.")
    parser.add_argument("--confirm-encryption", action="store_true", help="Required to modify stored private media and model blobs.")
    parser.add_argument("--rotate-existing", action="store_true", help="Decrypt with the active/previous key and re-encrypt all existing blobs with the active key.")
    args = parser.parse_args()
    if not args.confirm_encryption:
        raise SystemExit("This migration rewrites private database blobs. Back up the database first, then pass --confirm-encryption.")
    if not encryption_configured():
        raise SystemExit("Set FACE_DATA_ENCRYPTION_KEY before running this migration.")
    before = private_data_encryption_status()
    print("Plaintext blobs before migration:")
    print(json.dumps(before, indent=2))
    changed = encrypt_existing_private_data(reencrypt=args.rotate_existing)
    after = private_data_encryption_status()
    print("Rows encrypted:")
    print(json.dumps(changed, indent=2))
    print("Remaining plaintext blobs:")
    print(json.dumps(after, indent=2))
    if any(item["plaintext"] for item in after.values()):
        raise SystemExit("Migration incomplete. Keep the app in maintenance mode and investigate remaining plaintext rows.")
    print("Private data encryption migration completed.")


if __name__ == "__main__":
    main()
