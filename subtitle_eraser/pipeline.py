from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
import re
from threading import Lock, Thread

import cv2
import numpy as np
from .style import SubtitleStyle, style_color_mask

cv2.setNumThreads(2)


@dataclass
class TextRegion:
    polygon: np.ndarray
    text: str
    confidence: float = 1.0

    @property
    def chinese(self) -> bool:
        return bool(re.search(r"[\u3400-\u9fff\U00020000-\U0002fa1f]", self.text))

    @property
    def mixed(self) -> bool:
        return self.chinese and bool(re.search(r"[A-Za-z]", self.text))


@dataclass
class Settings:
    mode: str = "auto"
    font_color: str = "auto"
    edge_padding: int = 3
    confidence: float = 0.60
    detection_interval: float = 0.25
    repair_radius: int = 3
    style: SubtitleStyle | None = None
    # Runtime video scale relative to the selection; not a UI pixel setting.
    processing_scale: float = 1.0


@dataclass
class FrameResult:
    original: np.ndarray
    mask: np.ndarray
    repaired: np.ndarray
    regions: list[TextRegion]
    status: str
    timing: dict[str, float] = field(default_factory=dict)
    safe_to_present: bool = True

    def debug_image(self) -> np.ndarray:
        image = self.original.copy()
        selected = self.mask > 0
        image[selected] = (image[selected] * .30 + np.array([90, 220, 55]) * .70).astype(np.uint8)
        for region in self.regions:
            color = (70, 200, 100) if region.chinese and not region.mixed else (225, 175, 70)
            cv2.polylines(image, [region.polygon.astype(np.int32)], True, color, 1)
        return image


class OcrDetector:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        # The default 'min' expands a thin subtitle crop until its short side is
        # 736 px, wasting millions of pixels. Bound the long side instead.
        self.engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                               det_limit_type="max")

    def detect(self, image: np.ndarray) -> list[TextRegion]:
        # Limit our own input width: RapidOCR 1.4.4's 'max' preprocessing chooses
        # dynamic limits and does not enforce det_limit_side_len. Map polygons
        # using actual rounded dimensions, including downscaled wide displays.
        scale = min(max(1.0, min(2.0, 180 / image.shape[0])), 960 / image.shape[1])
        source = cv2.resize(image, None, fx=scale, fy=scale) if scale != 1 else image
        mapped_scale = np.array([source.shape[1]/image.shape[1], source.shape[0]/image.shape[0]], np.float32)
        result, _ = self.engine(source, use_cls=False)
        return [TextRegion(np.asarray(item[0], np.float32) / mapped_scale, item[1], float(item[2]))
                for item in (result or [])]


class AsyncOcrDetector:
    """One in-flight recognition, latest layout cache; never blocks video repair.

    Reset generations invalidate results from old crops or scenes. No frame queue
    is allowed to grow while recognition is slower than the video source.
    """
    def __init__(self):
        self.lock = Lock()
        self.generation = 0
        self.pending = False
        self.closed = False
        self.regions = []
        self.error = ""
        self.engine = None
        self.thread = None
        self.completed_at = 0.0
        self.recognition_ms = 0.0

    def reset(self):
        with self.lock:
            self.generation += 1
            self.regions = []
            self.error = ""

    def detect(self, image):
        with self.lock:
            if not self.pending and not self.closed:
                self.pending = True
                generation = self.generation
                self.thread = Thread(target=self._recognize, args=(image.copy(), generation), daemon=True)
                self.thread.start()
            if self.error:
                raise RuntimeError(self.error)
            # If recognition has failed to refresh for five seconds, preserve source pixels.
            return list(self.regions) if perf_counter()-self.completed_at < 5 else []

    def _recognize(self, image, generation):
        regions, error = [], ""
        started = perf_counter()
        try:
            if self.engine is None:
                self.engine = OcrDetector()
            regions = self.engine.detect(image)
        except Exception as exc:
            error = str(exc)
        with self.lock:
            if generation == self.generation and not self.closed:
                self.regions, self.error = regions, error
                self.completed_at = perf_counter()
                self.recognition_ms = (self.completed_at-started)*1000
            self.pending = False

    def close(self):
        with self.lock:
            self.closed = True
            self.regions = []


def polygon_mask(shape: tuple[int, int], regions: list[TextRegion], padding: int = 0) -> np.ndarray:
    result = np.zeros(shape, np.uint8)
    for region in regions:
        cv2.fillPoly(result, [np.rint(region.polygon).astype(np.int32)], 255)
    if padding:
        result = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (2 * padding + 1, 2 * padding + 1)))
    return result


def glyph_mask(image: np.ndarray, regions: list[TextRegion], settings: Settings) -> tuple[np.ndarray, str]:
    mask, status, _ = _glyph_mask_result(image, regions, settings)
    return mask, status


def stroke_candidates(image, settings, height=None):
    """Bright thin strokes before dilation; shared by repair and line fitting."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white = (hsv[:, :, 1] < 85) & (hsv[:, :, 2] > 170)
    yellow_saturation = 90 if settings.font_color == "yellow" else 140
    yellow = ((hsv[:, :, 0] >= 14) & (hsv[:, :, 0] <= 40) &
              (hsv[:, :, 1] > yellow_saturation) & (hsv[:, :, 2] > 145))
    color = white if settings.font_color == "white" else yellow if settings.font_color == "yellow" else white | yellow
    if settings.style is not None:
        color = style_color_mask(hsv, settings.style)
        height = settings.style.glyph_height or height
    kernel_size = int(np.clip((height or image.shape[0]) * .45, 9, 31)) | 1
    top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    seeds = (color & (top_hat > 18)).astype(np.uint8)*255
    return seeds, kernel_size


def estimate_text_scale(image, settings):
    # Actual glyph height supports a first selection made directly in fullscreen,
    # where there is no earlier windowed geometry to supply a scale factor.
    seeds, _ = stroke_candidates(image, settings)
    _, _, stats, _ = cv2.connectedComponentsWithStats(seeds, 8)
    w, h, area = stats[1:, 2], stats[1:, 3], stats[1:, 4]
    plausible = (h >= 8) & (h <= image.shape[0]*.97) & (w <= h*2.0) & (area < w*h*.85)
    heights = h[plausible]
    scale = max(1., min(4., float(np.percentile(heights, 80))/30)) if len(heights) >= 3 else 1.
    if scale <= 1.25:
        return 1.
    # Large lettering does not always have a thick outline. Require a dark,
    # neutral fringe beyond the normal 3px mask before enlarging its footprint.
    distance = cv2.distanceTransform(255-seeds, cv2.DIST_L2, 5)
    blue, green, red = cv2.split(image)
    maximum = cv2.max(cv2.max(blue, green), red)
    minimum = cv2.min(cv2.min(blue, green), red)
    dark = (maximum < 110) & (cv2.subtract(maximum, minimum) < 35)
    near = (distance > 3.5) & (distance <= max(4.5, 3.5*scale))
    far = (distance > 7*scale) & (distance <= 9*scale)
    if np.count_nonzero(near) < 20 or np.count_nonzero(far) < 20:
        return 1.
    near_dark, far_dark = np.mean(dark[near]), np.mean(dark[far])
    return scale if near_dark > .35 and near_dark-far_dark > .18 else 1.


def _glyph_mask_result(image: np.ndarray, regions: list[TextRegion], settings: Settings):
    """Conservative glyph refinement for common white/yellow, outlined subtitles.

    OCR boxes are gates, never the repair mask. English and ambiguous mixed-script
    boxes always take precedence. No previous-frame pixels are used as background.
    """
    shape = image.shape[:2]
    chinese = [r for r in regions if r.chinese and not r.mixed and r.confidence >= settings.confidence]
    protected = [r for r in regions if not r.chinese or r.mixed or r.confidence < settings.confidence]
    empty = np.zeros(shape, np.uint8)
    if not chinese:
        return empty, "中英混排：为保护英文已跳过" if any(r.mixed for r in regions) else "未检测到可处理的中文字幕", True

    allowed = polygon_mask(shape, chinese, settings.edge_padding + 2)
    guard = polygon_mask(shape, protected, settings.edge_padding + 3)
    heights = [max(r.polygon[:, 1]) - min(r.polygon[:, 1]) for r in chinese]
    seeds, kernel_size = stroke_candidates(image, settings, np.median(heights))
    seeds[(allowed == 0) | (guard > 0)] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, 8)
    # Index the component decision once per pixel. Re-scanning the whole crop for
    # every stroke is particularly costly for large subtitle rows.
    keep = ((stats[:, cv2.CC_STAT_AREA] >= 2) &
            (stats[:, cv2.CC_STAT_WIDTH] <= kernel_size * 8) &
            (stats[:, cv2.CC_STAT_HEIGHT] <= kernel_size * 5))
    keep[0] = False
    filtered = (keep[labels].astype(np.uint8) * 255)
    core_area = np.count_nonzero(filtered)
    if settings.edge_padding:
        p = settings.edge_padding
        filtered = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*p+1, 2*p+1)))
    filtered[(allowed == 0) | (guard > 0)] = 0
    # Reject gross segmentation failures instead of silently covering a full text box.
    allowed_area = max(1, np.count_nonzero(allowed))
    fraction = np.count_nonzero(filtered)/allowed_area
    # A tightly selected, thick outlined Chinese line legitimately occupies more
    # than 68% AFTER dilation. Judge the bright core too, instead of discarding it.
    if fraction > .90 or (fraction > .68 and core_area/allowed_area > .38):
        return empty, "文字轮廓不可靠，已保留原图；请缩小区域或调整字体颜色", False
    if not np.any(filtered):
        return empty, "未提取到匹配样式的文字轮廓，已保留原图", True
    return filtered, "局部修补中" + (" · 混排行已跳过" if any(r.mixed for r in regions) else ""), True


def repair(image: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    output = image.copy()
    points = cv2.findNonZero(mask)
    if points is None:
        return output
    x, y, w, h = cv2.boundingRect(points)
    margin = radius * 3 + 4
    x0, y0 = max(0, x-margin), max(0, y-margin)
    x1, y1 = min(image.shape[1], x+w+margin), min(image.shape[0], y+h+margin)
    part = cv2.inpaint(image[y0:y1, x0:x1], mask[y0:y1, x0:x1], radius, cv2.INPAINT_TELEA)
    selection = mask[y0:y1, x0:x1] > 0
    output[y0:y1, x0:x1][selection] = part[selection]
    return output


class SubtitleProcessor:
    def __init__(self, detector=None, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.detector = detector
        self.reset()

    def reset(self):
        self.regions: list[TextRegion] = []
        self.last_detected = -1e10
        self.last_thumb = None
        self.last_shape = None
        self.ocr_error = ""
        if hasattr(self.detector, "reset"):
            self.detector.reset()

    def process(self, image: np.ndarray, *, force_detection: bool = False) -> FrameResult:
        started = perf_counter()
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise ValueError("Expected a nonempty BGR frame")
        image = np.ascontiguousarray(image, dtype=np.uint8)
        if self.settings.mode == "manual":
            return self._manual_frame(image, started)
        thumb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (96, 32))
        scene_cut = (self.last_thumb is not None and
                     np.mean(cv2.absdiff(thumb, self.last_thumb)) > 30)
        if self.last_shape != image.shape or scene_cut:
            self.regions = []
            force_detection = True
            if hasattr(self.detector, "reset"):
                self.detector.reset()
        ocr_ms = 0.0
        if force_detection or started - self.last_detected >= self.settings.detection_interval:
            detect_start = perf_counter()
            try:
                if self.detector is None:
                    self.detector = OcrDetector()
                self.regions = self.detector.detect(image)
                self.ocr_error = ""
            except Exception as exc:
                self.regions = []
                self.ocr_error = f"识别失败，已保留原图：{type(exc).__name__}: {exc}"
            self.last_detected = perf_counter()
            ocr_ms = (self.last_detected - detect_start) * 1000
        mask, status, safe_to_present = _glyph_mask_result(image, self.regions, self.settings)
        if isinstance(self.detector, AsyncOcrDetector) and not self.regions and self.detector.pending:
            status = "正在后台定位字幕，暂时保留原图…"
        mask_ready = perf_counter()
        output = repair(image, mask, self.settings.repair_radius)
        done = perf_counter()
        self.last_thumb, self.last_shape = thumb, image.shape
        return FrameResult(image, mask, output, list(self.regions), self.ocr_error or status,
                           {"ocr_ms": ocr_ms, "mask_ms": (mask_ready-started)*1000-ocr_ms,
                            "repair_ms": (done-mask_ready)*1000, "total_ms": (done-started)*1000,
                            "background_ocr_ms": getattr(self.detector,"recognition_ms",0.0)},
                           safe_to_present=safe_to_present and not self.ocr_error)

    def _manual_frame(self, image, started):
        """The user owns the language boundary; never wait for OCR or scene cuts.

        Only use on a crop containing Chinese rows without English. The synthetic
        region is an allowed area, not an OCR result and not a rectangular mask.
        """
        scale = self.settings.processing_scale
        if scale <= 1.25:
            scale = max(scale, estimate_text_scale(image, self.settings))
        if scale > 1.25:
            return self._scaled_manual_frame(image, started, min(scale, 4.5))
        return self._manual_native_frame(image, started, self.settings)

    def _scaled_manual_frame(self, image, started, scale):
        """Repair enlarged subtitles at a bounded size, then restore only the mask.

        An extra working pixel covers resampling fringes around black outlines.
        All full-resolution background pixels outside the final mask stay exact.
        """
        h, w = image.shape[:2]
        size = (max(20, round(w/scale)), max(12, round(h/scale)))
        small = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        style = self.settings.style
        if style is not None and style.glyph_height is not None:
            style = replace(style, glyph_height=max(8, min(120, round(style.glyph_height/scale))))
        settings = replace(self.settings, style=style, processing_scale=1.,
                           edge_padding=self.settings.edge_padding + (1 if self.settings.edge_padding else 0))
        result = self._manual_native_frame(small, perf_counter(), settings)
        mask = cv2.resize(result.mask, (w, h), interpolation=cv2.INTER_NEAREST)
        if settings.edge_padding and np.any(mask):
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        output = image.copy()
        if np.any(mask):
            expanded = cv2.resize(result.repaired, (w, h), interpolation=cv2.INTER_LINEAR)
            output[mask > 0] = expanded[mask > 0]
        timing = {**result.timing, "total_ms": (perf_counter()-started)*1000, "working_scale": scale}
        return FrameResult(image, mask, output, [], result.status, timing, result.safe_to_present)

    def _manual_native_frame(self, image, started, settings):
        h, w = image.shape[:2]
        allowed = TextRegion(np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]],
                                      dtype=np.float32), "手动中文范围")
        mask, status, safe_to_present = _glyph_mask_result(image, [allowed], settings)
        mask_ready = perf_counter()
        output = repair(image, mask, settings.repair_radius)
        done = perf_counter()
        if np.any(mask):
            status = "低延迟修补中 · 按框选范围处理，不判断语言"
        elif "未提取" in status:
            status = "低延迟模式 · 当前无可处理轮廓"
        return FrameResult(image, mask, output, [], status,
                           {"ocr_ms": 0.0, "background_ocr_ms": 0.0,
                            "mask_ms": (mask_ready-started)*1000,
                            "repair_ms": (done-mask_ready)*1000,
                            "total_ms": (done-started)*1000}, safe_to_present=safe_to_present)
