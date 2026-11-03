"""PostgreSQL-backed integration checks for durable face-dataset jobs."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest

import studio_db


@pytest.fixture
def postgres_database(monkeypatch):
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        pytest.skip("A PostgreSQL DATABASE_URL is required for this integration test.")
    monkeypatch.setenv("DATABASE_URL", url)
    studio_db.ensure_schema()
    yield


def test_profile_photo_crud_and_training_data_reset(postgres_database):
    profile_id = uuid.uuid4().hex
    token = "photo_" + profile_id[:10]
    try:
        studio_db.create_face_profile_record({
            "id": profile_id,
            "name": "Photo CRUD " + profile_id[:8],
            "token": token,
            "description": "temporary photo metadata test",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "local_model_ready": True,
            "runpod_model_ready": True,
            "local_training_state": "completed",
            "runpod_training_state": "completed",
        })
        ungrouped_id = studio_db.add_face_photo(
            profile_id, "manual-left-45.jpg", b"manual-image",
            angle_label="Middle-left 45", photo_kind="manual",
            validation_status="not_checked",
        )
        group = studio_db.create_face_dataset_group(
            profile_id, "Angle examples", expected_count=1, dataset_type="face_angles"
        )
        grouped_id = studio_db.add_face_photo(
            profile_id, "grouped-right-90.jpg", b"grouped-image",
            group_id=group["id"], angle_label="Right profile",
            photo_kind="angle", validation_status="passed",
        )

        records = {item["id"]: item for item in studio_db.list_face_photo_metadata(profile_id)}
        assert set(records) == {ungrouped_id, grouped_id}
        assert "image_data" not in records[ungrouped_id]
        assert records[ungrouped_id]["group_id"] is None
        assert records[grouped_id]["group_id"] == group["id"]

        updated = studio_db.update_face_photo(
            ungrouped_id, angle_label="Middle-left 45°",
            training_caption="three-quarter left view",
            photo_kind="angle", validation_status="passed",
            validation_message="Manually verified.",
        )
        assert updated["photo_kind"] == "angle"
        assert updated["validation_status"] == "passed"
        assert updated["training_caption"] == "three-quarter left view"

        studio_db.save_face_model_bundle(profile_id, "local", "local-body.zip", b"local-adapter", 1)
        studio_db.save_face_model_bundle(profile_id, "runpod", "cloud-body.zip", b"cloud-adapter", 1)
        result = studio_db.clear_face_profile_training_data(profile_id)

        assert result == {"photos_deleted": 2, "groups_deleted": 1, "artifacts_deleted": 2}
        retained = studio_db.get_face_profile_record(profile_id)
        assert retained is not None
        assert retained["id"] == profile_id
        assert retained["image_count"] == 0
        assert retained["local_model_ready"] is False
        assert retained["runpod_model_ready"] is False
        assert retained["local_training_state"] == "not_trained"
        assert retained["runpod_training_state"] == "not_trained"
        assert studio_db.list_face_photo_metadata(profile_id) == []
        assert studio_db.list_face_dataset_groups(profile_id) == []
        assert studio_db.get_face_model_bundle(profile_id, "local") is None
        assert studio_db.get_face_model_bundle(profile_id, "runpod") is None
    finally:
        studio_db.delete_face_profile_record(profile_id)


def test_face_dataset_job_lifecycle_and_saved_asset_persistence(postgres_database):
    profile_id = uuid.uuid4().hex
    token = "test_" + profile_id[:10]
    asset_id = None
    try:
        profile = studio_db.create_face_profile_record({
            "id": profile_id,
            "name": "Integration " + profile_id[:8],
            "token": token,
            "description": "temporary integration-test profile",
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        assert profile and profile["id"] == profile_id

        # Simulate an older deployed schema whose required columns have no defaults.
        studio_db._run("ALTER TABLE face_dataset_groups ALTER COLUMN generated_count DROP DEFAULT")
        studio_db._run("ALTER TABLE face_dataset_groups ALTER COLUMN created_at DROP DEFAULT")
        studio_db._run("ALTER TABLE face_dataset_groups ALTER COLUMN updated_at DROP DEFAULT")
        group = studio_db.create_face_dataset_group(
            profile_id, "One-target integration batch", expected_count=1,
            dataset_type="face_angles", prompt="integration test",
        )
        assert group["generated_count"] == 0
        # The creation path should restore the missing default for older installs.
        defaults = studio_db._all(
            "SELECT column_name, column_default FROM information_schema.columns "
            "WHERE table_name='face_dataset_groups' "
            "AND column_name IN ('generated_count', 'created_at', 'updated_at')"
        )
        assert {row["column_name"] for row in defaults} == {"generated_count", "created_at", "updated_at"}
        assert all(row["column_default"] is not None for row in defaults)
        # Simulate the exact legacy-schema failure from the UI: face_photos.created_at
        # exists and is NOT NULL, but its default is missing.
        studio_db._run("ALTER TABLE face_photos ALTER COLUMN created_at DROP DEFAULT")
        source_id = studio_db.add_face_photo(
            profile_id, "source.jpg", b"source-image-placeholder",
            group_id=group["id"], angle_label="Original source",
            photo_kind="source", validation_status="not_applicable",
        )
        photo_timestamp = studio_db._one(
            "SELECT created_at FROM face_photos WHERE id=%s", (source_id,)
        )
        assert photo_timestamp and photo_timestamp["created_at"] is not None
        photo_default = studio_db._one(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_name='face_photos' AND column_name='created_at'"
        )
        assert photo_default and photo_default["column_default"] is not None
        job_id = uuid.uuid4().hex
        # Simulate the exact legacy failure: generated_count is NOT NULL but
        # has no default, alongside missing timestamp defaults.
        for column in (
            "generated_count", "current_index", "current_label", "group_count_base",
            "target_manifest", "cancel_requested", "failed_targets", "review_targets",
            "created_at", "updated_at",
        ):
            studio_db._run(
                f"ALTER TABLE face_dataset_jobs ALTER COLUMN {column} DROP DEFAULT"
            )
        queued = studio_db.create_face_dataset_job(
            job_id=job_id, group_id=group["id"], profile_id=profile_id,
            dataset_type="face_angles", backend="cpu", expected_count=1,
            source_photo_id=source_id,
            target_manifest=[{"label": "Front view", "yaw": 0.0, "pitch": 0.0}],
        )
        assert queued["status"] == "queued"
        assert queued["generated_count"] == 0
        assert queued["current_index"] == 0
        assert queued["current_label"] == ""
        assert queued["target_manifest"][0]["label"] == "Front view"
        job_columns = studio_db._all(
            "SELECT column_name, column_default FROM information_schema.columns "
            "WHERE table_name='face_dataset_jobs' AND column_name IN "
            "('generated_count', 'current_index', 'current_label', 'group_count_base', "
            "'target_manifest', 'cancel_requested', 'failed_targets', 'review_targets', "
            "'created_at', 'updated_at')"
        )
        assert len(job_columns) == 10
        assert all(row["column_default"] is not None for row in job_columns)

        running = studio_db.update_face_dataset_job(
            job_id, status="running", current_index=0, current_label="Front view",
        )
        assert running["status"] == "running"
        failed = studio_db.update_face_dataset_job(
            job_id, status="failed", error="RuntimeError: simulated worker failure",
        )
        assert failed["status"] == "failed"
        assert "simulated worker failure" in failed["error"]
        assert studio_db.get_face_dataset_job(job_id)["status"] == "failed"

        asset = studio_db.create_asset(
            asset_type="image", title="Saved source test", filename="saved-source.png",
            mime_type="image/png", media_data=b"saved-image-placeholder",
            prompt="integration source", backend="cpu", generation_mode="txt2img",
            generation_metadata={"seed": 1234, "steps": 25, "width": 512, "height": 768},
        )
        asset_id = asset["id"]
        restored = studio_db.get_asset(asset_id, include_content=True)
        assert restored["asset_type"] == "image"
        assert restored["media_data"] == b"saved-image-placeholder"
        assert restored["generation_metadata"]["seed"] == 1234
        assert restored["generation_metadata"]["width"] == 512
    finally:
        if asset_id:
            studio_db.delete_asset(asset_id)
        studio_db.delete_face_profile_record(profile_id)
