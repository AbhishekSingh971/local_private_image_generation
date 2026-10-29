"""Regression tests for failures while starting background face-dataset jobs."""
from __future__ import annotations

import io
import json
from types import SimpleNamespace

from PIL import Image

import app as studio_app


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_executor_start_failure_marks_job_and_group_failed(monkeypatch):
    profile_id = "b" * 32
    group_id = "c" * 32
    source_photo_id = "d" * 32
    source_asset_id = "a" * 32
    job_id = "e" * 32
    saved_image = _png_bytes()
    persisted_jobs = []
    persisted_groups = []

    monkeypatch.setattr(studio_app, "validate_profile_id", lambda value: value)
    monkeypatch.setattr(studio_app, "get_profile", lambda value: {"id": profile_id, "name": "Test profile"})
    monkeypatch.setattr(studio_app, "targets_for_dataset_type", lambda value: [
        {"label": "Front view", "yaw": 0.0, "pitch": 0.0, "prompt": "front view", "strength": 0.7}
    ])
    monkeypatch.setattr(studio_app, "_clean_prompt", lambda value, label, limit: value or "Test batch")
    monkeypatch.setattr(studio_app, "_resolve_compute_backend", lambda form: "cpu")
    monkeypatch.setattr(studio_app, "get_asset", lambda asset_id, include_content=False: {
        "id": source_asset_id, "asset_type": "image", "media_data": saved_image
    })
    monkeypatch.setattr(studio_app, "ensure_schema", lambda: None)
    monkeypatch.setattr(studio_app, "get_face_profile_record", lambda value: {"id": profile_id, "name": "Test profile"})
    monkeypatch.setattr(studio_app, "create_face_dataset_group", lambda *args, **kwargs: {"id": group_id, "profile_id": profile_id})
    monkeypatch.setattr(studio_app, "save_profile_image", lambda *args, **kwargs: SimpleNamespace(name="source.png"))
    monkeypatch.setattr(studio_app, "list_face_dataset_group_photo_records", lambda value: [
        {"id": source_photo_id, "filename": "source.png"}
    ])
    monkeypatch.setattr(studio_app, "create_face_dataset_job", lambda **kwargs: {
        "id": job_id, "group_id": group_id, "profile_id": profile_id,
        "dataset_type": "face_angles", "backend": "cpu", "status": "queued",
        "expected_count": 1, "generated_count": 0,
    })
    monkeypatch.setattr(studio_app, "get_face_dataset_group", lambda value: {
        "id": group_id, "profile_id": profile_id, "status": "generating"
    })
    monkeypatch.setattr(studio_app, "_create_face_dataset_job_payload", lambda job: dict(job))
    monkeypatch.setattr(
        studio_app, "update_face_dataset_job",
        lambda value, **fields: persisted_jobs.append((value, fields)) or {"id": value, **fields},
    )
    monkeypatch.setattr(
        studio_app, "update_face_dataset_group",
        lambda value, **fields: persisted_groups.append((value, fields)) or {"id": value, **fields},
    )
    monkeypatch.setattr(studio_app, "FACE_DATASET_JOB_EVENTS", {})
    monkeypatch.setattr(studio_app, "FACE_DATASET_JOB_FUTURES", {})

    class FailingPool:
        def submit(self, *args, **kwargs):
            raise RuntimeError("worker launch failed password=do-not-show")

    monkeypatch.setattr(studio_app, "FACE_DATASET_JOB_POOL", FailingPool())

    with studio_app.app.test_request_context(
        "/api/face-profiles/" + profile_id + "/dataset-jobs",
        method="POST",
        data={
            "dataset_type": "face_angles",
            "selected_targets": json.dumps(["Front view"]),
            "source_asset_id": source_asset_id,
            "title": "One target",
        },
    ):
        response, status = studio_app.api_start_face_dataset_job(profile_id)

    assert status == 503
    payload = response.get_json()
    assert "worker launch failed" in payload["error"]
    assert "do-not-show" not in payload["error"]
    assert persisted_jobs == [(job_id, {"status": "failed", "error": payload["error"]})]
    assert persisted_groups == [(group_id, {"status": "failed", "error": payload["error"]})]
    assert studio_app.FACE_DATASET_JOB_EVENTS == {}
    assert studio_app.FACE_DATASET_JOB_FUTURES == {}
    assert studio_app.FACE_DATASET_STARTING is False
