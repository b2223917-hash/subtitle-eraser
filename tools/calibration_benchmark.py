"""Compare default/manual and calibrated/manual synthetic processing.

Run from the project root: .venv/Scripts/python tools/calibration_benchmark.py
Timers do not measure screen capture or display latency. Ground truth chooses
one simulated click and scores output; it is never passed to the processor.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from subtitle_eraser.fixtures import SCENES, sample_frame
from subtitle_eraser.pipeline import Settings, SubtitleProcessor, polygon_mask
from subtitle_eraser.style import calibrate_style


class NoOcr:
    def detect(self, image):
        raise AssertionError("Manual benchmark unexpectedly invoked OCR")


def statistics(values):
    data = np.asarray(values, dtype=float)
    return {"p50": float(np.median(data)), "p95": float(np.percentile(data, 95)),
            "mean": float(data.mean()), "max": float(data.max())}


def score(result, source, clean, truth, english, elapsed):
    selected, true_pixels = result.mask > 0, truth > 0
    changed = np.any(source != result.repaired, axis=2)
    near = cv2.dilate(truth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    count = int(np.count_nonzero(true_pixels))
    return {"processing_ms": elapsed, **result.timing,
            "truth_pixels": count, "mask_pixels": int(np.count_nonzero(selected)),
            "intersection_pixels": int(np.count_nonzero(selected & true_pixels)),
            "background_selected_beyond_3px_margin": int(np.count_nonzero(selected & ~near)),
            "background_changed_beyond_3px_margin": int(np.count_nonzero(changed & ~near)),
            "outside_mask_changed_pixels": int(np.count_nonzero(changed & ~selected)),
            "english_changed_pixels": int(np.count_nonzero(changed & (english > 0))),
            "blank_frame_mask_pixels": int(np.count_nonzero(selected)) if not count else 0,
            "glyph_mae_before": float(np.abs(source[true_pixels].astype(float) - clean[true_pixels]).mean()) if count else 0,
            "glyph_mae_after": float(np.abs(result.repaired[true_pixels].astype(float) - clean[true_pixels]).mean()) if count else 0}


def summarize(rows):
    active = [row for row in rows if row["truth_pixels"]]
    truth = sum(row["truth_pixels"] for row in rows)
    sums = ("background_selected_beyond_3px_margin", "background_changed_beyond_3px_margin",
            "outside_mask_changed_pixels", "english_changed_pixels", "blank_frame_mask_pixels")
    return {"frames": len(rows), "visible_frames": len(active),
            "processing_ms_all_frames": statistics([r["processing_ms"] for r in rows]),
            "processing_ms_visible_subtitles": statistics([r["processing_ms"] for r in active]),
            "glyph_recall": sum(r["intersection_pixels"] for r in rows) / max(1, truth),
            "glyph_mae_before": float(np.mean([r["glyph_mae_before"] for r in active])),
            "glyph_mae_after": float(np.mean([r["glyph_mae_after"] for r in active])),
            "ocr_ms_total": sum(r["ocr_ms"] for r in rows),
            **{key: sum(r[key] for r in rows) for key in sums}}


def run_resolution(width, height, destination):
    roi = np.s_[int(height * .65):int(height * .82), int(width * .1):int(width * .9)]
    processors = {name: SubtitleProcessor(NoOcr(), Settings(mode="manual"))
                  for name in ("default", "calibrated")}
    records, calibrations = [], []
    for index in range(288):
        image, clean, truth, regions = sample_frame(index, width, height)
        english = polygon_mask(image.shape[:2], [r for r in regions if not r.chinese], 3)
        assert np.count_nonzero(truth[roi]) == np.count_nonzero(truth)
        assert not np.any(english[roi]), "English protection is a Chinese-only crop boundary"
        source, crop_truth = image[roi].copy(), truth[roi].copy()
        if index % 48 == 0:
            hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            stroke = ((crop_truth > 0) & (hsv[:, :, 2] >= 230)).astype(np.uint8)
            distance = cv2.distanceTransform(stroke, cv2.DIST_L2, 3)
            y, x = np.unravel_index(np.argmax(distance), distance.shape)
            start = perf_counter()
            style = calibrate_style(source, (int(x), int(y)))
            calibration_ms = (perf_counter() - start) * 1000
            processors["calibrated"].settings.style = style
            calibrations.append({"scene": SCENES[index // 48], "frame": index,
                                 "click_xy": [int(x), int(y)], "calibration_ms": calibration_ms,
                                 "style": style.to_dict()})
        row = {"frame": index, "scene": SCENES[index // 48]}
        # Alternate execution order to reduce a persistent first/second bias.
        order = ("default", "calibrated") if index % 2 == 0 else ("calibrated", "default")
        for name in order:
            start = perf_counter()
            result = processors[name].process(source)
            elapsed = (perf_counter() - start) * 1000
            row[name] = score(result, source, clean[roi], crop_truth, english[roi], elapsed)
        records.append(row)
        if index % 48 == 47:
            print(f"{width}x{height}: {index + 1}/288 source frames complete", flush=True)
    summaries = {}
    for name in processors:
        summaries[name] = {**summarize([row[name] for row in records]),
                           "scenes": [{"scene": scene, **summarize([row[name] for row in records if row["scene"] == scene])}
                                      for scene in SCENES]}
    (destination / f"frames-{width}x{height}.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"source_size": [width, height],
            "crop_xyxy": [roi[1].start, roi[0].start, roi[1].stop, roi[0].stop],
            "source_frames": 288, "processing_passes": 576,
            "calibration_ms": statistics([c["calibration_ms"] for c in calibrations]),
            "calibrations": calibrations, "modes": summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/calibration"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = {"generated_at": datetime.now().astimezone().isoformat(),
               "source": "deterministic synthetic Chinese/English fixtures; six scenes, 12 seconds, 24 fps per resolution; illustrated face, not real footage",
               "timing_scope": "CPU colour/stroke-mask and OpenCV inpainting only; excludes fixture creation, scoring, capture, GUI, overlay presentation and monitor scanout. These are NOT on-screen latency measurements.",
               "calibration_scope": "One simulated stroke click at the first frame of each scene. Calibration is measured separately and excluded from per-frame processing. Style/colour changes require recalibration; the yellow scene is not processed with a white style.",
               "quality_scope": "Ground truth is used only to choose a simulated click and score results. English lies entirely outside the Chinese-only crop. Glyph truth includes outlines; a 3px surrounding margin is allowed for mask dilation. This does not test real-video false-positive rates or language recognition.",
               "environment": {"python": platform.python_version(), "opencv": cv2.__version__,
                               "opencv_threads": cv2.getNumThreads(), "platform": platform.platform()},
               "runs": []}
    for width, height in ((960, 540), (1440, 810)):
        run = run_resolution(width, height, args.output)
        metrics["runs"].append(run)
        (args.output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"source_size": run["source_size"], "calibration_ms": run["calibration_ms"],
                          "modes": {name: {key: values[key] for key in ("processing_ms_visible_subtitles", "glyph_recall", "background_selected_beyond_3px_margin", "blank_frame_mask_pixels")}
                                    for name, values in run["modes"].items()}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
