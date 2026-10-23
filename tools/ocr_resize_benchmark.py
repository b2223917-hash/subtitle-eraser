"""Compare RapidOCR's default shortest-side resize with native-size detection.

Runs synthetic fixtures without opening a window. Timings are local observations,
not end-to-end display latency or claims about arbitrary videos.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr_onnxruntime import RapidOCR

from subtitle_eraser.fixtures import SCENES, sample_frame
from subtitle_eraser.pipeline import Settings, TextRegion, glyph_mask, polygon_mask, repair


def run(repeats=2):
    records = []
    x, y, w, h = 76, 324, 808, 194
    for mode in ("min", "max"):
        engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                          det_limit_type=mode)
        session = engine.text_det.infer.session
        opts = session.get_session_options()
        for scene, index in enumerate((12, 60, 108, 156, 204, 252)):
            frame, clean, truth, reference = sample_frame(index)
            crop = frame[y:y+h, x:x+w]
            scale = max(1.0, min(2.0, 180 / h))
            source = cv2.resize(crop, None, fx=scale, fy=scale) if scale > 1 else crop
            prep, _, _ = engine.preprocess(source)
            prep, _ = engine.maybe_add_letterbox(prep, {})
            tensor = engine.text_det.get_preprocess(max(prep.shape[:2]))(prep)
            times = []
            for _ in range(repeats):
                start = perf_counter()
                result, parts = engine(source, use_cls=False)
                times.append({"total_ms": (perf_counter()-start)*1000,
                              "parts_ms": [value*1000 for value in (parts or [])]})
            regions = [TextRegion(np.asarray(item[0], np.float32)/scale,
                                  item[1], float(item[2])) for item in (result or [])]
            mask, status = glyph_mask(crop, regions, Settings())
            repaired = repair(crop, mask)
            english = polygon_mask(frame.shape[:2], [r for r in reference if not r.chinese], 2)[y:y+h, x:x+w] > 0
            actual = truth[y:y+h, x:x+w] > 0
            changed = np.any(crop != repaired, axis=2)
            row = {"mode": mode, "scene": SCENES[scene], "frame_index": index,
                   "source_shape_hwc": list(source.shape), "detector_tensor_nchw": list(tensor.shape),
                   "detector_providers": session.get_providers(),
                   "intra_op_num_threads": opts.intra_op_num_threads,
                   "inter_op_num_threads": opts.inter_op_num_threads,
                   "timings": times,
                   "recognized": [{"text": r.text, "confidence": r.confidence} for r in regions],
                   "chinese_glyph_recall": float(np.count_nonzero(actual & (mask > 0))/max(1, np.count_nonzero(actual))),
                   "english_changed_pixels": int(np.count_nonzero(changed & english)),
                   "outside_mask_changed_pixels": int(np.count_nonzero(changed & (mask == 0))),
                   "status": status}
            records.append(row)
            print(f"{mode} scene {scene+1}: {times[-1]['total_ms']:.1f} ms; recall {row['chinese_glyph_recall']:.4f}", flush=True)
    return {"created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source": "six synthetic frames; known Chinese glyph masks and English guard boxes",
            "roi_xywh": [x,y,w,h], "repeats_per_scene": repeats,
            "available_providers": ort.get_available_providers(),
            "notes": ["Timing includes detection and recognition, excludes model construction and UI/capture latency.",
                      "First invocation can include warmup; raw repetitions retained. No real-video accuracy claim.",
                      "RapidOCR 1.4.4 max mode chooses its own max-side threshold; det_limit_side_len alone is not a hard maximum."],
            "official_default_config": "https://raw.githubusercontent.com/RapidAI/RapidOCR/v1.4.4/python/rapidocr_onnxruntime/config.yaml",
            "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/low-latency/ocr_resize_metrics.json"))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    metrics = run(args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output.resolve())
