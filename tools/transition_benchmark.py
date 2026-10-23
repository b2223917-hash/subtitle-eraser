"""Reproducible overlay timing simulation, NOT a desktop latency measurement.

The source changes independently from a processed crop arriving 40 ms later.
Known synthetic glyph masks isolate presentation leakage from recognition or
segmentation failures. OpenCV repairs both modes with exactly the same pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from subtitle_eraser.fixtures import load_font
from subtitle_eraser.pipeline import repair
from subtitle_eraser.presentation import overlay_rgba

WIDTH, HEIGHT = 640, 256
# The user-selected crop contains Chinese only; English remains below it.
ROI = (24, 124, 592, 62)
STEP_MS, DELAY_MS, SWITCH_MS = 10, 40, 250
SCENARIOS = {
    "short_to_long": ("旧句", "字幕更新后的长句出现了"),
    "blank_to_new": ("", "新字幕出现了"),
}


def crop(image):
    x, y, width, height = ROI
    return image[y:y + height, x:x + width]


def make_source(scenario: str, time_ms: int):
    """Return current BGR source and exact Chinese glyph support, no OCR."""
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    background = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    for channel, base in enumerate((58, 43, 32)):
        background[:, :, channel] = base + ((xx + yy // 2) // 40) % 10
    # A moving, dark object makes the whole-crop background delay visible too.
    moving_x = int((time_ms * .32 + 120) % (WIDTH + 80)) - 40
    cv2.rectangle(background, (moving_x, 80), (moving_x + 38, 194), (100, 70, 45), -1)
    canvas = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(canvas)
    glyphs = Image.new("L", (WIDTH, HEIGHT))
    glyph_draw = ImageDraw.Draw(glyphs)
    text = SCENARIOS[scenario][time_ms >= SWITCH_MS]
    if text:
        font = load_font(26)
        bounds = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        left = (WIDTH - (bounds[2] - bounds[0])) // 2
        location = (left, 130)
        draw.text(location, text, font=font, fill=(248, 248, 248),
                  stroke_width=2, stroke_fill=(12, 12, 12))
        glyph_draw.text(location, text, font=font, fill=255,
                        stroke_width=2, stroke_fill=255)
    draw.text((88, 204), "Keep the English. Enjoy the video.", font=load_font(23),
              fill=(248, 248, 248), stroke_width=2, stroke_fill=(12, 12, 12))
    source = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    support = (np.asarray(glyphs) > 0).astype(np.uint8) * 255
    return source, support


def compose(source: np.ndarray, rgba: np.ndarray):
    """Composite just the selected crop over the current, independent source."""
    result = source.copy()
    selected = crop(result)
    alpha = rgba[:, :, 3:4].astype(np.float32) / 255
    repaired_bgr = rgba[:, :, :3][:, :, ::-1]
    selected[:] = np.rint(repaired_bgr * alpha + selected * (1 - alpha)).astype(np.uint8)
    return result


def simulate_frame(scenario: str, time_ms: int, delay_ms=DELAY_MS):
    source, current_support = make_source(scenario, time_ms)
    prior, prior_support = make_source(scenario, time_ms - delay_ms)
    known_mask = cv2.dilate(crop(prior_support), np.ones((3, 3), np.uint8))
    # Important: BOTH overlay modes receive repaired pixels, never raw subtitles.
    repaired = repair(crop(prior), known_mask)
    glyph_rgba = overlay_rgba(repaired, known_mask)
    crop_rgba = overlay_rgba(repaired, known_mask, cover_region=True)
    current_chinese = crop(current_support) > 0
    row = {
        "source_ms": time_ms,
        "processed_source_ms": time_ms - delay_ms,
        "current_chinese_pixels": int(current_chinese.sum()),
        "glyph_only_exposed_source_pixels": int(np.count_nonzero(current_chinese & (glyph_rgba[:, :, 3] == 0))),
        "opaque_crop_exposed_source_pixels": int(np.count_nonzero(current_chinese & (crop_rgba[:, :, 3] == 0))),
        "opaque_crop_min_alpha": int(crop_rgba[:, :, 3].min()),
        "prior_mask_empty": not bool(known_mask.any()),
    }
    return source, compose(source, glyph_rgba), compose(source, crop_rgba), row


def labeled(image, title, subtitle):
    panel = cv2.copyMakeBorder(image, 58, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 30, 39))
    cv2.putText(panel, title, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, .58, (240, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, subtitle, (14, 45), cv2.FONT_HERSHEY_SIMPLEX, .43, (175, 190, 205), 1, cv2.LINE_AA)
    x, y, width, height = ROI
    cv2.rectangle(panel, (x - 1, y + 57), (x + width, y + height + 58), (170, 140, 60), 1)
    return panel


def comparison_row(scenario, time_ms):
    raw, old, new, metrics = simulate_frame(scenario, time_ms)
    panels = [
        labeled(raw, "SOURCE / " + scenario, f"Source t={time_ms} ms; change at {SWITCH_MS} ms"),
        labeled(old, "OLD: GLYPH-ONLY ALPHA", f"Exposed new glyph pixels: {metrics['glyph_only_exposed_source_pixels']}"),
        labeled(new, "NEW: OPAQUE PROCESSED CROP", f"Exposed new glyph pixels: {metrics['opaque_crop_exposed_source_pixels']}"),
    ]
    combined = np.concatenate(panels, axis=1)
    combined = cv2.copyMakeBorder(combined, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 30, 39))
    cv2.putText(combined, "SOFTWARE SIMULATION: assumed 40 ms delay; exact glyph masks; NOT a screen measurement. Video plays 4x slower.",
                (14, 21), cv2.FONT_HERSHEY_SIMPLEX, .48, (140, 205, 245), 1, cv2.LINE_AA)
    return combined, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "transition")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "software_compositing_timing_simulation",
        "processing_delay_ms_assumed": DELAY_MS,
        "simulation_step_ms": STEP_MS,
        "source_switch_ms": SWITCH_MS,
        "roi_xywh": ROI,
        "repair": "OpenCV Telea with exact synthetic glyph support, plus one-pixel dilation",
        "limitations": [
            "This is an assumed-delay software simulation, not a real screen or end-to-end latency measurement.",
            "It isolates raw source leakage through transparent alpha. It does not measure OCR, missed glyphs, or repair quality.",
            "The opaque crop uses older background pixels; moving content can reveal its boundary.",
            "Continuity starts after a first repaired crop is available. Alignment, focus changes, and capture failures are not simulated.",
            "comparison.mp4 plays the simulated 10 ms steps at 25 fps (four times slower).",
        ],
        "scenarios": {},
    }
    snapshots = []
    video_frames = []
    for scenario in SCENARIOS:
        samples = []
        outside_changed = english_changed = 0
        outside = np.ones((HEIGHT, WIDTH), bool)
        crop(outside)[:] = False
        for time_ms in range(0, 500, STEP_MS):
            raw, old, new, sample = simulate_frame(scenario, time_ms)
            samples.append(sample)
            outside_changed += int(np.count_nonzero(np.any(raw[outside] != new[outside], axis=1)))
            english_changed += int(np.count_nonzero(np.any(raw[200:244] != new[200:244], axis=2)))
            if not args.no_video:
                panel, _ = comparison_row(scenario, time_ms)
                video_frames.append(panel)
        snapshot, _ = comparison_row(scenario, SWITCH_MS)
        snapshots.append(snapshot)
        report["scenarios"][scenario] = {
            "glyph_only_source_exposure_ms": sum(s["glyph_only_exposed_source_pixels"] > 0 for s in samples) * STEP_MS,
            "opaque_crop_source_exposure_ms": sum(s["opaque_crop_exposed_source_pixels"] > 0 for s in samples) * STEP_MS,
            "glyph_only_peak_exposed_pixels": max(s["glyph_only_exposed_source_pixels"] for s in samples),
            "opaque_crop_peak_exposed_pixels": max(s["opaque_crop_exposed_source_pixels"] for s in samples),
            "all_opaque_crop_alpha_255": all(s["opaque_crop_min_alpha"] == 255 for s in samples),
            "outside_crop_changed_pixels": outside_changed,
            "english_changed_pixels": english_changed,
            "samples": samples,
        }
    combined = np.concatenate(snapshots, axis=0)
    cv2.imencode(".png", combined)[1].tofile(args.output / "comparison.png")
    if video_frames:
        import imageio_ffmpeg
        target = args.output / "comparison.mp4"
        height, width = video_frames[0].shape[:2]
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                   "-r", "25", "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in video_frames:
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Could not encode comparison video")
    (args.output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in data.items() if key != "samples"}
                      for name, data in report["scenarios"].items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
