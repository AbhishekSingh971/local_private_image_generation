"""Purpose-aware dataset selection for identity and body LoRA training.

Unknown legacy files are retained only for ordinary identity profiles. They are
never silently treated as body-training examples.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

PURPOSE_KINDS = {
    "identity": {"source", "manual"},
    "body": {"body"},
    "expression": {"expression"},
    "angle": {"angle"},
}


def select_training_files(
    files,
    metadata_by_filename: Mapping[str, Mapping] | None,
    purpose: str,
    *,
    include_unclassified_identity: bool = True,
):
    purpose = (purpose or "").strip().lower()
    if purpose not in PURPOSE_KINDS:
        raise ValueError("Training purpose must be identity, body, expression or angle.")
    metadata = metadata_by_filename or {}
    selected, excluded = [], []
    allowed = PURPOSE_KINDS[purpose]
    for item in sorted((Path(path) for path in files), key=lambda path: path.name):
        record = metadata.get(item.name)
        if record is None:
            keep = purpose == "identity" and include_unclassified_identity
            reason = "legacy_unclassified_identity" if keep else "missing_purpose_metadata"
        else:
            kind = str(record.get("photo_kind") or "manual").strip().lower()
            status = str(record.get("validation_status") or "not_checked").strip().lower()
            keep = kind in allowed and status != "needs_review"
            reason = "selected" if keep else (
                "requires_human_review" if status == "needs_review" else f"wrong_purpose:{kind}"
            )
        if keep:
            selected.append(item)
        else:
            excluded.append({"filename": item.name, "reason": reason})
    return selected, excluded
