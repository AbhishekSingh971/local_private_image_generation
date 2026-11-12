"""Strict file selection for human-approved face model training datasets."""
from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable


def select_approved_training_files(
    dataset_dir: str | Path,
    approved_filenames: Iterable[str],
    allowed_extensions: set[str],
) -> list[Path]:
    """Select only exact approved basenames and fail if any approved file is missing."""
    names = list(approved_filenames)
    if len(names) != len(set(names)):
        raise ValueError("Approved training photo filenames must be unique.")

    safe_names: set[str] = set()
    for value in names:
        name = str(value or "")
        path = Path(name)
        if not name or path.name != name or name in {".", ".."}:
            raise ValueError("Approved training photo filenames must be plain file basenames.")
        if path.suffix.lower() not in allowed_extensions:
            raise ValueError(f"Unsupported approved training photo extension: {path.suffix or '(none)'}")
        safe_names.add(name)

    folder = Path(dataset_dir)
    found = {
        path.name: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    } if folder.is_dir() else {}
    missing = safe_names - set(found)
    if missing:
        raise ValueError(
            "One or more approved training photos are missing from the profile dataset: "
            + ", ".join(sorted(missing))
        )
    return [found[name] for name in sorted(safe_names)]
