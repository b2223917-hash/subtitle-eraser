"""Measure manual Chinese-only crops; synthetic processing, not screen latency.

Fixture creation, media encoding and metric calculations are excluded from the
processing timer. No OCR model, GUI or capture session is started.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subtitle_eraser.fixtures import SCENES, sample_frame
from subtitle_eraser.pipeline import Settings, SubtitleProcessor, polygon_mask


class NoOcr:
    def detect(self, image):
        raise AssertionError("Manual benchmark unexpectedly invoked OCR")


def stats(values):
    data = np.asarray(values, dtype=float)
    return {"p50": float(np.median(data)), "p95": float(np.percentile(data, 95)),
            "mean": float(data.mean()), "max": float(data.max())}


def video_panel(image, title):
    width = 600
    height = int(round(image.shape[0] * width / image.shape[1] / 2)) * 2
    panel = cv2.resize(image, (width, height))
    panel = cv2.copyMakeBorder(panel, 32, 0, 0, 0, cv2.BORDER_CONSTANT, value=(28, 34, 43))
    cv2.putText(panel, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, .52,
                (235, 240, 250), 1, cv2.LINE_AA)
    return panel


def run_resolution(width, height, destination, media=False):
    fps, frame_count = 24, 288
    x0, x1 = int(width * .10), int(width * .90)
    y0, y1 = int(height * .65), int(height * .82)
    roi = (slice(y0, y1), slice(x0, x1))
    prepared = []
    # Retain only the narrow crops, avoiding gigabytes of full-frame fixtures.
    for index in range(frame_count):
        image, clean, truth, regions = sample_frame(index, width, height, fps)
        english = polygon_mask(image.shape[:2], [r for r in regions if not r.chinese], 3)
        assert np.count_nonzero(truth[roi]) == np.count_nonzero(truth)
        prepared.append((image[roi].copy(), clean[roi].copy(), truth[roi].copy(), english[roi].copy()))

    processor = SubtitleProcessor(NoOcr(), Settings(mode="manual"))
    records, media_frames = [], []
    for index, (source, clean, truth, english) in enumerate(prepared):
        start = perf_counter()
        result = processor.process(source)
        elapsed_ms = (perf_counter() - start) * 1000
        truth_selected = truth > 0
        selected = result.mask > 0
        changed = np.any(source != result.repaired, axis=2)
        near_glyph = cv2.dilate(truth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
        truth_count = int(np.count_nonzero(truth_selected))
        background_count = int(np.count_nonzero(~truth_selected))
        error_before = float(np.abs(source[truth_selected].astype(float) - clean[truth_selected]).mean()) if truth_count else 0.0
        error_after = float(np.abs(result.repaired[truth_selected].astype(float) - clean[truth_selected]).mean()) if truth_count else 0.0
        record = {"frame": index, "scene": SCENES[index // 48], "processing_ms": elapsed_ms,
                  **result.timing, "truth_pixels": truth_count,
                  "mask_pixels": int(np.count_nonzero(selected)),
                  "intersection_pixels": int(np.count_nonzero(selected & truth_selected)),
                  "background_pixels": background_count,
                  "background_changed_pixels": int(np.count_nonzero(changed & ~truth_selected)),
                  "background_changed_beyond_3px_margin": int(np.count_nonzero(changed & ~near_glyph)),
                  "english_changed_pixels": int(np.count_nonzero(changed & (english > 0))),
                  "outside_mask_changed_pixels": int(np.count_nonzero(changed & ~selected)),
                  "blank_frame_mask_pixels": int(np.count_nonzero(selected)) if not truth_count else 0,
                  "glyph_mae_before": error_before, "glyph_mae_after": error_after}
        records.append(record)
        if media:
            frame = np.concatenate([video_panel(source, "Original Chinese-only crop"),
                                    video_panel(result.debug_image(), "Glyph mask (green)"),
                                    video_panel(result.repaired, "Local background repair")], axis=1)
            media_frames.append(frame)

    by_scene = defaultdict(list)
    for record in records:
        by_scene[record["scene"]].append(record)
    scenes = []
    for scene, rows in by_scene.items():
        active = [r for r in rows if r["truth_pixels"]]
        total_truth = sum(r["truth_pixels"] for r in rows)
        scenes.append({"scene": scene, "frames": len(rows), "visible_frames": len(active),
                       "processing_ms": stats([r["processing_ms"] for r in rows]),
                       "glyph_recall": sum(r["intersection_pixels"] for r in rows) / max(1, total_truth),
                       "background_changed_pixels": sum(r["background_changed_pixels"] for r in rows),
                       "background_changed_fraction": sum(r["background_changed_pixels"] for r in rows) / sum(r["background_pixels"] for r in rows),
                       "background_changed_beyond_3px_margin": sum(r["background_changed_beyond_3px_margin"] for r in rows),
                       "english_changed_pixels": sum(r["english_changed_pixels"] for r in rows),
                       "outside_mask_changed_pixels": sum(r["outside_mask_changed_pixels"] for r in rows),
                       "blank_frame_mask_pixels": sum(r["blank_frame_mask_pixels"] for r in rows),
                       "glyph_mae_before": float(np.mean([r["glyph_mae_before"] for r in active])),
                       "glyph_mae_after": float(np.mean([r["glyph_mae_after"] for r in active]))})
    data = {"source_size": [width, height], "crop_xyxy": [x0, y0, x1, y1],
            "crop_size": [x1 - x0, y1 - y0], "frames": frame_count,
            "first_frame_processing_ms": records[0]["processing_ms"],
            "first_frame_mask_pixels": records[0]["mask_pixels"],
            "processing_ms_all_frames": stats([r["processing_ms"] for r in records]),
            "processing_ms_visible_subtitles": stats([r["processing_ms"] for r in records if r["truth_pixels"]]),
            "ocr_ms_total": sum(r["ocr_ms"] for r in records), "scenes": scenes}
    (destination / f"frames-{width}x{height}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    if media:
        from imageio_ffmpeg import get_ffmpeg_exe
        # Encode after all timing measurements. Preview plays at source rate, not
        # measured processing rate, and deliberately shows only the manual crop.
        comparison = destination / "comparison.mp4"
        video_height, video_width = media_frames[0].shape[:2]
        command = [get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                   "-vcodec", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{video_width}x{video_height}",
                   "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
                   "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        try:
            for frame in media_frames:
                process.stdin.write(frame.tobytes())
        finally:
            process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("Comparison encoding failed")
        sheet = np.concatenate([media_frames[i * 48] for i in range(6)], axis=0)
        cv2.imencode(".png", sheet)[1].tofile(destination / "comparison.png")
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/low-latency"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {"generated_at": datetime.now().astimezone().isoformat(),
               "source": "deterministic synthetic fixtures only; six scenes, 12 seconds, 24 fps",
               "mode": "manual, user-selected Chinese-only crop; no OCR",
               "timing_scope": "CPU mask generation and OpenCV repair only; excludes input generation, capture, UI, encoding, display scanout. First-frame processing is not first on-screen repair latency.",
               "quality_scope": "English is protected by being outside the selected crop, not by language detection. Exact glyph truth includes outline; background_changed_pixels includes the necessary mask edge padding. Beyond-3px counts report changes outside that padding allowance. Faces are illustrations, not real footage.",
               "preview_scope": "comparison.mp4 plays source frames at 24 fps, not a recording of live processing",
               "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                               "opencv_threads": cv2.getNumThreads(), "platform": platform.platform()},
               "runs": []}
    for index, (width, height) in enumerate(((960, 540), (1440, 810))):
        result = run_resolution(width, height, args.output, media=(index == 0))
        metrics["runs"].append(result)
        (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"source_size": result["source_size"],
                          "first_frame_processing_ms": result["first_frame_processing_ms"],
                          "active_frame_ms": result["processing_ms_visible_subtitles"],
                          "glyph_recall": [s["glyph_recall"] for s in result["scenes"]]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
