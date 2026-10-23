"""Portable-build startup check. Does not automate the desktop or capture user data."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from PySide6.QtWidgets import QApplication
import numpy as np
import cv2
from .fixtures import sample_frame
from .pipeline import SubtitleProcessor, Settings
from .capture import WindowCapture
from .ui import MainWindow, qimage
from .style import calibrate_style, SubtitleStyle
from .profiles import ProfileStore
from .windows import player_identity


def run(output: Path):
    app=QApplication.instance() or QApplication([])
    profile_directory = TemporaryDirectory(prefix="subtitle-selftest-")
    profile_path = Path(profile_directory.name) / "profiles.json"
    window=MainWindow(profile_store=ProfileStore(profile_path))
    image,_,_,_=sample_frame(0)
    result=SubtitleProcessor().process(image[324:518,76:884],force_detection=True)
    assert any(r.chinese for r in result.regions),"Chinese OCR unavailable"
    assert any(not r.chinese for r in result.regions),"English OCR unavailable"
    assert np.any(result.mask),"No glyph mask"
    assert qimage(result.repaired).width()==808
    window.overlay.set_result(result.repaired,result.mask)
    assert window.overlay.image.hasAlphaChannel()
    manual = SubtitleProcessor(settings=Settings(mode="manual"))
    immediate = manual.process(image[351:443,96:864])
    assert manual.detector is None, "Manual mode must not load OCR"
    assert np.any(immediate.mask), "Manual mode must repair the first frame"
    assert np.array_equal(immediate.original[immediate.mask == 0], immediate.repaired[immediate.mask == 0])
    assert window.settings().mode == "manual"
    assert window.cover_region(), "Continuous crop presentation must default on"
    window.overlay.set_result(immediate.repaired, immediate.mask, cover_region=True)
    alpha = np.frombuffer(window.overlay.image.constBits(), dtype=np.uint8).reshape(
        window.overlay.image.height(), window.overlay.image.bytesPerLine())[:, 3::4]
    assert np.all(alpha == 255), "Processed crop must be opaque, including gaps"
    window.overlay.set_result(immediate.repaired, np.zeros_like(immediate.mask), cover_region=True)
    alpha = np.frombuffer(window.overlay.image.constBits(), dtype=np.uint8).reshape(
        window.overlay.image.height(), window.overlay.image.bytesPerLine())[:, 3::4]
    assert np.all(alpha == 255), "Blank subtitle interval must stay covered"
    face, _, face_truth, _ = sample_frame(96)
    face_result = manual.process(face[351:443,96:864])
    edge_allowance = cv2.dilate(face_truth[351:443,96:864],
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    assert not np.any((face_result.mask > 0) & (edge_allowance == 0)), "Skin mistaken for yellow text"
    crop = image[351:443,96:864]
    bright = np.argwhere(np.all(crop >= 240, axis=2))
    py, px = bright[len(bright)//2]
    style = calibrate_style(crop, (int(px), int(py)))
    calibrated = SubtitleProcessor(settings=Settings(mode="manual", style=style)).process(crop)
    assert np.any(calibrated.mask) and calibrated.safe_to_present
    assert SubtitleStyle.from_dict(style.to_dict()) == style
    identity = player_identity(int(window.winId()))
    assert identity, "Player executable lookup unavailable"
    assert window.profile_store.save(identity, (.1, .65, .8, .17), (960, 540),
        {"mode": "manual", "font_color": "auto", "edge_padding": 3,
         "style": style.to_dict(), "continuity": True})
    restored = ProfileStore(profile_path).lookup(identity, (960, 540))
    assert restored["settings"]["style"] == style.to_dict()
    from .diagnostics import native_geometry_check
    native_checks = native_geometry_check()
    data={"passed":True,"qt":True,"native_capture_import":True,"ocr_models":True,
          "manual_mode":True,"manual_first_frame_ms":immediate.timing["total_ms"],
          "manual_background_guard":True,
          "continuous_region":True,
          "style_calibration":True, "profile_roundtrip":True, "player_identity":True,
          "alpha_overlay":True,"native_overlay_geometry":native_checks,"mask_pixels":int(np.count_nonzero(result.mask)),
          "timing":result.timing,"recognized":[r.text for r in result.regions]}
    window.close()
    profile_directory.cleanup()
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")
    return 0
