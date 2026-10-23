import json

import pytest

from subtitle_eraser.profiles import ProfileStore, default_path


IDENTITY = r"C:\Users\Private Name\Apps\Player.exe"
ROI = (.1, .7, .8, .1)
SIZE = (1920, 1080)
SETTINGS = {"mode": "manual", "font_color": "white", "edge_padding": 2,
            "style": None, "continuity": True}


def test_round_trip_is_private_and_lookup_is_defensive(tmp_path):
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    assert store.save(IDENTITY, ROI, SIZE, SETTINGS)
    text = path.read_text("utf-8")
    assert "Private Name" not in text and "Users" not in text
    data = json.loads(text)
    assert len(next(iter(data["profiles"]))) == 64
    restored = ProfileStore(path)
    assert restored.load()
    profile = restored.lookup(IDENTITY.upper(), SIZE)
    assert profile == {"app": "player.exe", "roi": list(ROI), "frame_size": list(SIZE), "settings": SETTINGS}
    profile["settings"]["font_color"] = "yellow"
    profile["roi"][0] = .9
    assert restored.lookup(IDENTITY, SIZE)["settings"] == SETTINGS
    assert restored.lookup(IDENTITY, SIZE)["roi"] == list(ROI)


def test_read_only_and_missing_profile_do_not_create_files(tmp_path):
    path = tmp_path / "nested" / "profiles.json"
    store = ProfileStore(path)
    assert store.load()
    assert store.lookup(IDENTITY, SIZE) is None
    assert store.remove(IDENTITY)
    assert not path.parent.exists()


def test_frame_size_change_rejects_restore(tmp_path):
    store = ProfileStore(tmp_path / "profiles.json")
    assert store.save(IDENTITY, ROI, SIZE, SETTINGS)
    assert store.lookup(IDENTITY, (1280, 720)) is None
    assert "尺寸" in store.last_error
    assert store.lookup(IDENTITY, SIZE) is not None
    assert not store.last_error


@pytest.mark.parametrize("roi", [(-.1, .1, .5, .1), (.5, .1, .6, .1), (0, 0, 0, 1),
                                  (0, 0, float("nan"), 1), (0, 0, True, 1), (0, 0, 1)])
def test_invalid_roi_does_not_write(tmp_path, roi):
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    assert not store.save(IDENTITY, roi, SIZE, SETTINGS)
    assert store.last_error and not path.exists()


@pytest.mark.parametrize("settings", [{"password": "secret"}, {"mode": "anything"},
    {"edge_padding": True}, {"edge_padding": 6}, {"edge_padding": 99}, {"continuity": "yes"}, {"font_color": []}])
def test_unknown_or_invalid_settings_are_rejected(tmp_path, settings):
    store = ProfileStore(tmp_path / "profiles.json")
    assert not store.save(IDENTITY, ROI, SIZE, settings)
    assert store.last_error


def test_corrupt_config_is_not_overwritten(tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text("broken", encoding="utf-8")
    store = ProfileStore(path)
    assert not store.load()
    assert not store.save(IDENTITY, ROI, SIZE, SETTINGS)
    assert not store.remove(IDENTITY)
    assert path.read_text("utf-8") == "broken"


def test_invalid_disk_profile_rejects_whole_file(tmp_path):
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    assert store.save(IDENTITY, ROI, SIZE, SETTINGS)
    data = json.loads(path.read_text("utf-8"))
    next(iter(data["profiles"].values()))["title"] = "Private video title"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert not ProfileStore(path).load()


def test_atomic_failure_preserves_previous_file_and_memory(tmp_path, monkeypatch):
    import subtitle_eraser.profiles as profiles
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    assert store.save(IDENTITY, ROI, SIZE, SETTINGS)
    before = path.read_bytes()
    def denied(*args):
        raise PermissionError("sharing violation")
    monkeypatch.setattr(profiles.os, "replace", denied)
    assert not store.save(IDENTITY, (.2, .7, .5, .1), SIZE, SETTINGS)
    assert path.read_bytes() == before
    assert store.lookup(IDENTITY, SIZE)["roi"] == list(ROI)
    assert not list(tmp_path.glob(".profiles-*.tmp"))


def test_remove_persists_without_touching_other_players(tmp_path):
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    assert store.save(IDENTITY, ROI, SIZE, SETTINGS)
    assert store.save("other.exe", ROI, SIZE, SETTINGS)
    assert store.remove(IDENTITY)
    restored = ProfileStore(path)
    assert restored.lookup(IDENTITY, SIZE) is None
    assert restored.lookup("other.exe", SIZE) is not None


def test_default_uses_local_app_data_without_creating_files(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert default_path() == tmp_path / "SubtitleEraser" / "profiles.json"
    assert not default_path().exists()


def test_unwritable_parent_reports_error(tmp_path):
    parent = tmp_path / "file"
    parent.write_text("existing")
    store = ProfileStore(parent / "profiles.json")
    assert not store.save(IDENTITY, ROI, SIZE, SETTINGS)
    assert store.last_error and parent.read_text() == "existing"


def test_calibrated_style_round_trips_as_numeric_values_only(tmp_path):
    from subtitle_eraser.style import SubtitleStyle
    path = tmp_path / "profiles.json"
    style = SubtitleStyle(30, 225, 250, glyph_height=28, edge_padding=2)
    settings = dict(SETTINGS, style=style.to_dict())
    assert ProfileStore(path).save(IDENTITY, ROI, SIZE, settings)
    restored = ProfileStore(path).lookup(IDENTITY, SIZE)
    assert SubtitleStyle.from_dict(restored["settings"]["style"]) == style


@pytest.mark.parametrize("style", [{"hue": 0, "saturation": 0, "value": 255, "screenshot": "data"},
    {"hue": 0, "saturation": 0, "value": 255, "glyph_height": True}, {"hue": 0}, []])
def test_invalid_calibrated_style_is_not_persisted(tmp_path, style):
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    assert not store.save(IDENTITY, ROI, SIZE, dict(SETTINGS, style=style))
    assert store.last_error and not path.exists()


def test_partial_settings_receive_canonical_defaults(tmp_path):
    path = tmp_path / "profiles.json"
    assert ProfileStore(path).save(IDENTITY, ROI, SIZE, {"font_color": "yellow"})
    expected = {"mode": "manual", "font_color": "yellow", "edge_padding": 3,
                "style": None, "continuity": True}
    assert ProfileStore(path).lookup(IDENTITY, SIZE)["settings"] == expected
    # Older manually edited profiles may omit fields; load uses the same schema.
    data = json.loads(path.read_text("utf-8"))
    next(iter(data["profiles"].values()))["settings"] = {}
    path.write_text(json.dumps(data), encoding="utf-8")
    expected["font_color"] = "auto"
    assert ProfileStore(path).lookup(IDENTITY, SIZE)["settings"] == expected
