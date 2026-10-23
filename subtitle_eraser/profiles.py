"""Small local player profiles. No titles, screenshots, or executable paths on disk."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import ntpath
import os
from pathlib import Path
import re
import tempfile


_MAX_BYTES = 1024 * 1024
_MAX_PROFILES = 128


def default_path() -> Path:
    root = os.environ.get("LOCALAPPDATA")
    return (Path(root) if root else Path.home() / "AppData" / "Local") / "SubtitleEraser" / "profiles.json"


def _identity(identity: str) -> tuple[str, str]:
    if not isinstance(identity, str) or not identity.strip() or len(identity) > 32768:
        raise ValueError("播放器标识无效")
    normalized = ntpath.normpath(identity.strip()).casefold()
    app = ntpath.basename(normalized)
    if not app or any(ord(c) < 32 for c in app) or len(app) > 255:
        raise ValueError("播放器名称无效")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest(), app


def _size(value) -> list[int]:
    if (not isinstance(value, (list, tuple)) or len(value) != 2 or
            any(type(v) is not int or not 1 <= v <= 32768 for v in value)):
        raise ValueError("画面尺寸无效")
    return list(value)


def _roi(value) -> list[float]:
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            any(type(v) not in (float, int) or not math.isfinite(v) for v in value)):
        raise ValueError("字幕区域无效")
    x, y, width, height = (float(v) for v in value)
    if (not 0 <= x < 1 or not 0 <= y < 1 or not 0 < width <= 1 or not 0 < height <= 1 or
            x + width > 1.000000001 or y + height > 1.000000001):
        raise ValueError("字幕区域超出画面")
    return [x, y, min(width, 1 - x), min(height, 1 - y)]


def _settings(value) -> dict:
    allowed = {"mode", "font_color", "edge_padding", "style", "continuity"}
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise ValueError("播放器设置包含未知项目")
    result = {"mode": "manual", "font_color": "auto", "edge_padding": 3,
              "style": None, "continuity": True}
    result.update(deepcopy(value))
    if "mode" in value and value["mode"] not in ("auto", "manual"):
        raise ValueError("处理方式无效")
    if "font_color" in value and value["font_color"] not in ("auto", "white", "yellow"):
        raise ValueError("字幕颜色无效")
    if "edge_padding" in value and (type(value["edge_padding"]) is not int or not 0 <= value["edge_padding"] <= 5):
        raise ValueError("修补边缘无效")
    if "continuity" in value and type(value["continuity"]) is not bool:
        raise ValueError("防露字设置无效")
    if value.get("style") is not None:
        from .style import SubtitleStyle
        result["style"] = asdict(SubtitleStyle.from_dict(value["style"]))
    return result


def _profile(value) -> dict:
    if not isinstance(value, dict) or set(value) != {"app", "roi", "frame_size", "settings"}:
        raise ValueError("保存的播放器配置格式无效")
    app = value["app"]
    if (not isinstance(app, str) or not app or len(app) > 255 or ntpath.basename(app) != app or
            any(ord(c) < 32 for c in app)):
        raise ValueError("保存的播放器名称无效")
    return {"app": app, "roi": _roi(value["roi"]), "frame_size": _size(value["frame_size"]),
            "settings": _settings(value["settings"])}


class ProfileStore:
    """Operations return success/None and set last_error instead of raising in the UI.

    ``lookup`` only returns a profile for the exact captured frame size. The caller
    must ask the user to confirm the restored rectangle on a current preview.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else default_path()
        self.last_error = ""
        self._profiles: dict[str, dict] = {}
        self._loaded = False
        self._load_ok = False

    def load(self) -> bool:
        self.last_error = ""
        self._loaded = True
        try:
            if not self.path.exists():
                self._profiles = {}
                self._load_ok = True
                return True
            if self.path.stat().st_size > _MAX_BYTES:
                raise ValueError("配置文件过大")
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(data, dict) or set(data) != {"version", "profiles"} or
                    type(data["version"]) is not int or data["version"] != 1 or
                    not isinstance(data["profiles"], dict) or len(data["profiles"]) > _MAX_PROFILES):
                raise ValueError("配置版本或格式无效")
            profiles = {}
            for key, value in data["profiles"].items():
                if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                    raise ValueError("播放器索引无效")
                profiles[key] = _profile(value)
            self._profiles = profiles
            self._load_ok = True
            return True
        except (OSError, ValueError, TypeError, ImportError) as exc:
            self._load_ok = False
            self.last_error = f"无法读取播放器设置：{exc}"
            return False

    def _ready(self) -> bool:
        if not self._loaded:
            return self.load()
        if not self._load_ok:
            self.last_error = "播放器设置尚未成功读取；请先检查配置文件，避免覆盖已有设置"
            return False
        return True

    def lookup(self, identity: str, frame_size) -> dict | None:
        if not self._ready():
            return None
        self.last_error = ""
        try:
            key, _ = _identity(identity)
            size = _size(frame_size)
            profile = self._profiles.get(key)
            if profile is None:
                return None
            if profile["frame_size"] != size:
                self.last_error = "画面尺寸与保存时不同，请重新框选字幕区域"
                return None
            return deepcopy(profile)
        except (ValueError, TypeError) as exc:
            self.last_error = str(exc)
            return None

    def save(self, identity: str, roi, frame_size, settings: dict) -> bool:
        if not self._ready():
            return False
        self.last_error = ""
        try:
            key, app = _identity(identity)
            value = _profile({"app": app, "roi": roi, "frame_size": frame_size, "settings": settings})
            profiles = deepcopy(self._profiles)
            profiles[key] = value
            if len(profiles) > _MAX_PROFILES:
                raise ValueError("已保存的播放器过多，请先删除不用的设置")
            return self._write(profiles)
        except (ValueError, TypeError, ImportError) as exc:
            self.last_error = f"无法保存播放器设置：{exc}"
            return False

    def remove(self, identity: str) -> bool:
        if not self._ready():
            return False
        self.last_error = ""
        try:
            key, _ = _identity(identity)
            if key not in self._profiles:
                return True
            profiles = deepcopy(self._profiles)
            del profiles[key]
            return self._write(profiles)
        except (ValueError, TypeError) as exc:
            self.last_error = f"无法删除播放器设置：{exc}"
            return False

    def _write(self, profiles: dict) -> bool:
        temporary = None
        try:
            payload = json.dumps({"version": 1, "profiles": profiles}, ensure_ascii=False, indent=2, allow_nan=False)
            if len(payload.encode("utf-8")) > _MAX_BYTES:
                raise ValueError("配置文件过大")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=".profiles-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._profiles = profiles
            return True
        except (OSError, ValueError, TypeError) as exc:
            self.last_error = f"无法保存播放器设置：{exc}"
            return False
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
