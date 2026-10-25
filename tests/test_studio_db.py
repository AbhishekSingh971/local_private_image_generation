from datetime import datetime, timezone
import uuid

import studio_db


def test_assets_and_face_models_are_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "studio.sqlite").as_posix())
    studio_db.ensure_schema()
    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "Studio A",
        "token": "studio_a_" + profile_id[:5],
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    studio_db.add_face_photo(profile_id, "face_test.jpg", b"photo-bytes")
    asset = studio_db.create_asset(
        asset_type="image", title="Portrait draft", filename="portrait.png",
        mime_type="image/png", media_data=b"png-bytes", prompt="a portrait",
        negative_prompt="blurry", profile_id=profile_id, profile_name="Studio A",
        backend="cpu", generation_mode="txt2img",
    )
    assert studio_db.get_face_profile_record(profile_id)["image_count"] == 1
    assert studio_db.list_face_photo_names(profile_id) == {"face_test.jpg"}
    assert studio_db.get_asset(asset["id"], include_content=True)["media_data"] == b"png-bytes"
    backup = studio_db.save_face_model_bundle(profile_id, "local", "model.zip", b"weights", 1)
    assert backup["bundle_size"] == len(b"weights")
    assert studio_db.get_face_model_bundle(profile_id, "local")["bundle_data"] == b"weights"
    assert [item["id"] for item in studio_db.list_assets("image")] == [asset["id"]]
    assert studio_db.list_assets("video") == []
    updated = studio_db.update_asset(asset["id"], {"title": "Portrait final"})
    assert updated["title"] == "Portrait final"
    assert studio_db.delete_asset(asset["id"])["filename"] == "portrait.png"
    assert studio_db.get_asset(asset["id"]) is None


def test_database_status_does_not_disclose_database_path(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "status.sqlite").as_posix())
    status = studio_db.database_status()
    assert status["connected"] is True
    assert status["configured_postgresql"] is False
    assert "status.sqlite" not in status["message"]
