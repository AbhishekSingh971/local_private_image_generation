from datetime import datetime, timezone
import uuid

import studio_db
import face_profiles


def test_assets_and_face_models_are_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "studio.sqlite").as_posix())
    studio_db.ensure_schema()
    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "Studio A",
        "token": "studio_a_" + profile_id[:5],
        "description": "Short curly hair and brown eyes",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    group = studio_db.create_face_dataset_group(profile_id, "First angle set", expected_count=5)
    photo_id = studio_db.add_face_photo(
        profile_id, "face_test.jpg", b"photo-bytes", group_id=group["id"], angle_label="Original source"
    )
    asset = studio_db.create_asset(
        asset_type="image", title="Portrait draft", filename="portrait.png",
        mime_type="image/png", media_data=b"png-bytes", prompt="a portrait",
        negative_prompt="blurry", profile_id=profile_id, profile_name="Studio A",
        backend="cpu", generation_mode="txt2img",
    )
    stored_profile = studio_db.get_face_profile_record(profile_id)
    assert stored_profile["image_count"] == 1
    assert stored_profile["description"] == "Short curly hair and brown eyes"
    assert studio_db.list_face_photo_names(profile_id) == {"face_test.jpg"}
    stored_photo, photo_bytes = studio_db.get_face_photo_bytes(photo_id)
    assert photo_bytes == b"photo-bytes"
    assert stored_photo["group_id"] == group["id"]
    assert stored_photo["angle_label"] == "Original source"
    listed_groups = studio_db.list_face_dataset_groups(profile_id)
    assert len(listed_groups) == 1
    assert listed_groups[0]["photos"][0]["id"] == photo_id
    assert listed_groups[0]["photo_count"] == 1
    completed_group = studio_db.update_face_dataset_group(
        group["id"], status="completed", generated_count=5
    )
    assert completed_group["status"] == "completed"
    assert completed_group["generated_count"] == 5
    assert studio_db.get_asset(asset["id"], include_content=True)["media_data"] == b"png-bytes"
    backup = studio_db.save_face_model_bundle(profile_id, "local", "model.zip", b"weights", 1)
    assert backup["bundle_size"] == len(b"weights")
    assert studio_db.get_face_model_bundle(profile_id, "local")["bundle_data"] == b"weights"
    assert [item["id"] for item in studio_db.list_assets("image")] == [asset["id"]]
    assert studio_db.count_assets("image") == 1
    assert studio_db.count_assets("video") == 0
    assert studio_db.list_assets("image", limit=1, offset=1) == []
    assert studio_db.list_assets("image", search="portrait")[0]["id"] == asset["id"]
    assert studio_db.list_assets("image", search="not-a-match") == []
    assert studio_db.list_assets("video") == []
    updated = studio_db.update_asset(asset["id"], {"title": "Portrait final"})
    assert updated["title"] == "Portrait final"
    assert studio_db.delete_asset(asset["id"])["filename"] == "portrait.png"
    assert studio_db.get_asset(asset["id"]) is None


def test_dataset_group_and_photo_crud(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "dataset-crud.sqlite").as_posix())
    studio_db.ensure_schema()
    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "CRUD Model",
        "token": "crud_" + profile_id[:6],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    group = studio_db.create_face_dataset_group(profile_id, "Initial batch", expected_count=5)
    source_id = studio_db.add_face_photo(
        profile_id, "source.jpg", b"source-bytes", group_id=group["id"],
        angle_label="Original source", photo_kind="source",
    )
    angle_id = studio_db.add_face_photo(
        profile_id, "front.jpg", b"front-bytes", group_id=group["id"],
        angle_label="Front view", photo_kind="angle",
    )

    renamed = studio_db.update_face_dataset_group(group["id"], title="Renamed batch")
    assert renamed["title"] == "Renamed batch"
    relabeled = studio_db.update_face_photo(angle_id, angle_label="Straight-on view")
    assert relabeled["angle_label"] == "Straight-on view"
    assert relabeled["photo_kind"] == "angle"

    listed = studio_db.list_face_dataset_groups(profile_id)
    assert listed[0]["photo_count"] == 2
    assert {photo["id"] for photo in listed[0]["photos"]} == {source_id, angle_id}

    deleted_photo = studio_db.delete_face_photo(angle_id)
    assert deleted_photo["id"] == angle_id
    assert studio_db.get_face_photo_bytes(angle_id) is None
    assert studio_db.list_face_dataset_groups(profile_id)[0]["photo_count"] == 1

    # Deleting the last photo must not silently delete its saved dataset group.
    deleted_source = studio_db.delete_face_photo(source_id)
    assert deleted_source["id"] == source_id
    empty_groups = studio_db.list_face_dataset_groups(profile_id)
    assert len(empty_groups) == 1
    assert empty_groups[0]["id"] == group["id"]
    assert empty_groups[0]["photo_count"] == 0
    assert studio_db.get_face_dataset_group(group["id"]) is not None

    # The group is removed only after the user explicitly deletes the group.
    deleted_group = studio_db.delete_face_dataset_group(group["id"])
    assert deleted_group["id"] == group["id"]
    assert deleted_group["photos"] == []
    assert studio_db.get_face_dataset_group(group["id"]) is None
    assert studio_db.list_face_dataset_groups(profile_id) == []


def test_database_status_does_not_disclose_database_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "status.sqlite").as_posix())
    status = studio_db.database_status()
    assert status["connected"] is True
    assert status["configured_postgresql"] is False
    assert "status.sqlite" not in status["message"]



def test_legacy_profile_migration_does_not_resurrect_deleted_profile(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    profiles_dir = data_dir / "face_profiles"
    model_dir = tmp_path / "models" / "face_loras"
    profile_id = "a" * 32
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "migration.sqlite").as_posix())
    monkeypatch.setattr(face_profiles, "DATA_DIR", data_dir)
    monkeypatch.setattr(face_profiles, "PROFILES_FILE", data_dir / "face_profiles.json")
    monkeypatch.setattr(face_profiles, "PROFILES_DIR", profiles_dir)
    monkeypatch.setattr(face_profiles, "LORA_DIR", model_dir)
    monkeypatch.setattr(face_profiles, "_DB_MIGRATED", False)
    dataset = profiles_dir / profile_id / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "face_legacy.jpg").write_bytes(b"legacy-photo")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "face_profiles.json").write_text(
        '{"version": 1, "profiles": [{"id": "' + profile_id + '", "name": "Legacy Face", '
        '"token": "face_legacy", "created_at": "2026-01-01T00:00:00+00:00"}]}',
        encoding="utf-8",
    )

    assert len(face_profiles.list_profiles()) == 1
    assert studio_db.get_face_profile_record(profile_id)["image_count"] == 1

    # Simulate loss of the local cache; the database should rebuild it on a new process start.
    (dataset / "face_legacy.jpg").unlink()
    monkeypatch.setattr(face_profiles, "_DB_MIGRATED", False)
    assert len(face_profiles.list_profiles()) == 1
    assert (dataset / "face_legacy.jpg").read_bytes() == b"legacy-photo"

    assert face_profiles.delete_profile(profile_id) is True

    # A new process would run the migration again; the persisted marker prevents resurrection.
    monkeypatch.setattr(face_profiles, "_DB_MIGRATED", False)
    assert face_profiles.list_profiles() == []



def test_typed_dataset_groups_persist_category_and_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "typed-groups.sqlite").as_posix())
    studio_db.ensure_schema()
    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "Typed Model",
        "token": "typed_" + profile_id[:6],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })

    default_group = studio_db.create_face_dataset_group(profile_id, "Angles")
    assert default_group["dataset_type"] == "face_angles"
    assert default_group["prompt"] == ""
    assert default_group["generated_count"] == 0

    expression_group = studio_db.create_face_dataset_group(
        profile_id, "Expressions", dataset_type="expressions", prompt="Keep this person's identity and vary expression."
    )
    body_group = studio_db.create_face_dataset_group(
        profile_id, "Shared body", expected_count=5, dataset_type="body_type",
        prompt="Photorealistic adult model with a naturally athletic build.",
    )
    assert expression_group["dataset_type"] == "expressions"
    assert expression_group["prompt"] == "Keep this person's identity and vary expression."
    assert body_group["dataset_type"] == "body_type"
    assert "athletic build" in body_group["prompt"]

    listed = {group["id"]: group for group in studio_db.list_face_dataset_groups(profile_id)}
    assert listed[expression_group["id"]]["dataset_type"] == "expressions"
    assert listed[body_group["id"]]["prompt"] == body_group["prompt"]



def test_background_dataset_job_state_and_pose_validation_persist(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "background-job.sqlite").as_posix())
    studio_db.ensure_schema()
    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "Background Model",
        "token": "background_" + profile_id[:5],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    group = studio_db.create_face_dataset_group(
        profile_id, "Nine angle targets", expected_count=9, dataset_type="face_angles",
    )
    source_id = studio_db.add_face_photo(
        profile_id, "source.jpg", b"source-bytes", group_id=group["id"],
        angle_label="Original source", photo_kind="source", validation_status="not_applicable",
    )
    photo_id = studio_db.add_face_photo(
        profile_id, "left-profile.jpg", b"generated-bytes", group_id=group["id"],
        angle_label="Left profile · Needs review", photo_kind="angle",
        validation_status="needs_review",
        validation_message="Estimated yaw did not match the target.",
        training_caption="head pose: yaw -90 degrees, pitch 0 degrees",
        target_yaw=-90.0, target_pitch=0.0, actual_yaw=-8.0, actual_pitch=1.0,
    )
    job_id = uuid.uuid4().hex
    manifest = [
        {"label": "Left profile", "yaw": -90.0, "pitch": 0.0},
        {"label": "Right profile", "yaw": 90.0, "pitch": 0.0},
    ]
    job = studio_db.create_face_dataset_job(
        job_id=job_id, group_id=group["id"], profile_id=profile_id,
        dataset_type="face_angles", backend="cpu", expected_count=2,
        source_photo_id=source_id, target_manifest=manifest, group_count_base=7,
    )
    assert job["status"] == "queued"
    assert job["expected_count"] == 2
    assert job["target_manifest"] == manifest
    assert job["group_count_base"] == 7
    running = studio_db.update_face_dataset_job(
        job_id, status="running", current_index=6, current_label="Left profile",
        current_task_id=uuid.uuid4().hex, generated_count=5,
        review_targets=[{"label": "Left profile", "photo_id": photo_id}],
    )
    assert running["status"] == "running"
    assert running["generated_count"] == 5
    assert running["target_manifest"] == manifest
    assert running["group_count_base"] == 7
    assert running["review_targets"][0]["label"] == "Left profile"
    assert studio_db.get_face_dataset_job(job_id)["current_index"] == 6
    assert [row["id"] for row in studio_db.list_face_dataset_jobs(active_only=True)] == [job_id]

    stored_meta, stored_bytes = studio_db.get_face_photo_bytes(photo_id)
    assert stored_bytes == b"generated-bytes"
    assert stored_meta["validation_status"] == "needs_review"
    assert stored_meta["target_yaw"] == -90.0
    assert stored_meta["training_caption"].startswith("head pose:")

    listed = studio_db.list_face_dataset_groups(profile_id)[0]
    listed_photo = next(photo for photo in listed["photos"] if photo["id"] == photo_id)
    assert listed_photo["validation_status"] == "needs_review"
    assert "did not match" in listed_photo["validation_message"]

    cancelled = studio_db.update_face_dataset_job(job_id, status="cancelling", cancel_requested=True)
    assert cancelled["cancel_requested"] is True
    assert cancelled["status"] == "cancelling"
    cancelled = studio_db.update_face_dataset_job(job_id, status="cancelled", current_task_id=None)
    assert cancelled["status"] == "cancelled"
    assert studio_db.list_face_dataset_jobs(active_only=True) == []



def test_update_face_photo_validation_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "pose-update.sqlite").as_posix())
    studio_db.ensure_schema()
    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "Pose Metadata",
        "token": "pose_" + profile_id[:7],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    group = studio_db.create_face_dataset_group(profile_id, "Angles", expected_count=9)
    photo_id = studio_db.add_face_photo(
        profile_id, "pose.jpg", b"pose-bytes", group_id=group["id"],
        angle_label="Middle-left", photo_kind="angle",
        target_yaw=-30.0, target_pitch=0.0,
    )
    result = studio_db.update_face_photo_validation(
        photo_id, validation_status="passed", validation_message="Pose is within tolerance.",
        actual_yaw=-27.0, actual_pitch=2.0,
    )
    assert result["validation_status"] == "passed"
    stored, _ = studio_db.get_face_photo_bytes(photo_id)
    assert stored["actual_yaw"] == -27.0
    assert stored["validation_message"] == "Pose is within tolerance."



def test_replacing_photo_resets_pose_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "replace-validation.sqlite").as_posix())
    studio_db.ensure_schema()
    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "Replace Review",
        "token": "replace_" + profile_id[:5],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    group = studio_db.create_face_dataset_group(profile_id, "Angles", expected_count=9)
    photo_id = studio_db.add_face_photo(
        profile_id, "middle-left.jpg", b"original", group_id=group["id"],
        angle_label="Middle-left", photo_kind="angle",
        validation_status="passed", validation_message="Pose verified.",
        target_yaw=-30.0, target_pitch=0.0, actual_yaw=-32.0, actual_pitch=1.0,
    )
    updated = studio_db.update_face_photo(photo_id, image_data=b"edited")
    assert updated["validation_status"] == "needs_review"
    metadata, stored = studio_db.get_face_photo_bytes(photo_id)
    assert stored == b"edited"
    assert metadata["validation_status"] == "needs_review"
    assert metadata["actual_yaw"] is None
    assert "replaced" in metadata["validation_message"]
