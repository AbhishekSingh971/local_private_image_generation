"""Safe, versioned face-model artifact utilities."""
from __future__ import annotations

import hashlib
import io
import shutil
import tempfile
import zipfile
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_extract_lora_bundle(bundle_data: bytes, destination: str | Path) -> list[str]:
    """Extract a LoRA ZIP while rejecting traversal, symlinks and unexpected payloads."""
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(io.BytesIO(bundle_data)) as archive:
            members = archive.infolist()
            if not members or len(members) > 500:
                raise ValueError("Model bundle contains an invalid number of files.")
            seen_names = set()
            for member in members:
                name = member.filename.replace("\\", "/")
                if name in seen_names:
                    raise ValueError("Model bundle contains duplicate file paths.")
                seen_names.add(name)
                relative = Path(name)
                mode = (member.external_attr >> 16) & 0xFFFF
                if (
                    not name or relative.is_absolute()
                    or any(part in {"", ".", ".."} for part in relative.parts)
                    or ":" in relative.parts[0]
                    or (mode and (mode & 0o170000) == 0o120000)
                ):
                    raise ValueError("Model bundle contains an unsafe file path or link.")
                if member.is_dir():
                    continue
                total_uncompressed += int(member.file_size)
                if total_uncompressed > 2 * 1024 * 1024 * 1024:
                    raise ValueError("Model bundle expands beyond the 2 GB safety limit.")
                output = target.joinpath(*relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, output.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                written.append(relative.as_posix())
    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
        raise ValueError("Model bundle is corrupt or could not be extracted safely.") from exc

    if not any(Path(name).name in {"pytorch_lora_weights.safetensors", "pytorch_lora_weights.bin"} for name in written):
        raise ValueError("Restored bundle does not contain a supported face LoRA weight file.")
    return written
