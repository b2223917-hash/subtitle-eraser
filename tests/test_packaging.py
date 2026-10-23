from pathlib import Path
import json
import subprocess
import os
import pytest

DIST_ROOTS = ([os.environ["SUBTITLE_ERASER_DIST_ROOT"]] if os.environ.get("SUBTITLE_ERASER_DIST_ROOT") else
              ["dist", "dist-low-latency", "dist-stable", "dist-calibrated"])


@pytest.mark.parametrize("dist_root", DIST_ROOTS)
def test_portable_loads_qt_and_models_from_another_working_directory(tmp_path, dist_root):
    root=Path(__file__).resolve().parent.parent
    exe=root/dist_root/"SubtitleEraser"/"SubtitleEraser.exe"
    if not exe.exists():
        pytest.skip("Portable build not present")
    output=tmp_path/"selftest.json"
    completed=subprocess.run([str(exe),"--self-test","--output",str(output)],cwd=tmp_path,
                             timeout=30,creationflags=0x08000000)
    assert completed.returncode==0
    data=json.loads(output.read_text(encoding="utf-8"))
    assert data["qt"] and data["ocr_models"] and data["alpha_overlay"]
    assert data["mask_pixels"]>0
    if dist_root != "dist":
        assert data["manual_mode"] and data["manual_background_guard"]
    if dist_root not in ("dist", "dist-low-latency"):
        assert data["continuous_region"]
    if dist_root not in ("dist", "dist-low-latency", "dist-stable"):
        assert data["style_calibration"] and data["profile_roundtrip"] and data["player_identity"]


@pytest.mark.parametrize("dist_root", DIST_ROOTS)
def test_portable_icu_has_not_been_collected(dist_root):
    root=Path(__file__).resolve().parent.parent
    bundle=root/dist_root/"SubtitleEraser"/"_internal"
    if bundle.exists():
        assert not (bundle/"icuuc.dll").exists(),"Foreign ICU can prevent Qt startup"
