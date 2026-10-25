from datetime import datetime, timezone
import uuid

import studio_db


def test_assets_are_stored_and_crud_is_database_backed(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "studio.sqlite").as_posix())
    studio_db.ensure_schema()

    profile_id = uuid.uuid4().hex
    studio_db.create_face_profile_record({
        "id": profile_id,
        "name": "Studio A",
        "token": "studio_a",
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    studio_db.add_face_photo(profile_id, "face_test.jpg", b"photo-bytes")

    saved = studio_db.create_asset(
        asset_type="image",
        title="Portrait draft",
        filename="portrait.png",
        mime_type="image/png",
        media_data=b"png-bytes",
        prompt="a portrait",
        negative_prompt="blurry",
        profile_id=profile_id,
        profile_name="Studio A",
        backend="cpu",
        generation_mode="txt2img",
    )
    assert saved["media_size"] == len(b"png-bytes")
    assert studio_db.get_face_profile_record(profile_id)["image_count"] == 1
    assert studio_db.list_face_photo_names(profile_id) == {"face_test.jpg"}

    stored = studio_db.get_asset(saved["id"], include_content=True)
    assert stored["media_data"] == b"png-bytes"
    assert [row["id"] for row in studio_db.list_assets("image")] == [saved["id"]]
    assert studio_db.list_assets("video") == []

    changed = studio_db.update_asset(saved["id"], {"title": "Portrait final"})
    assert changed["title"] == "Portrait final"
    assert studio_db.get_asset(saved["id"])["title"] == "Portrait final"

    deleted = studio_db.delete_asset(saved["id"])
    assert deleted["filename"] == "portrait.png"
    assert studio_db.get_asset(saved["id"]) is None


def test_database_status_never_exposes_connection_string(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///" + (tmp_path / "status.sqlite").as_posix())
    result = studio_db.database_status()
    assert result["connected"] is True
    assert result["configured_postgresql"] is True  # an explicit DATABASE_URL was supplied
    assert "status.sqlite" not in result["message"]
