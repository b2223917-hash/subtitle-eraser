"""Small, local subtitle colour calibration; never persists image pixels.

A click identifies a stroke, not a language or a text box. The caller must still
limit processing to the user-selected Chinese subtitle area and refine this
colour mask with stroke/shape checks before inpainting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


def _integer(value: Any, name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} 必须是 {lower}–{upper} 之间的整数")
    return value


@dataclass(frozen=True)
class SubtitleStyle:
    """HSV uses OpenCV's H=0..179, S/V=0..255 representation."""

    hue: int
    saturation: int
    value: int
    color_tolerance: int = 24
    glyph_height: int | None = None
    edge_padding: int = 3

    def __post_init__(self):
        _integer(self.hue, "色相", 0, 179)
        _integer(self.saturation, "饱和度", 0, 255)
        _integer(self.value, "亮度", 145, 255)
        _integer(self.color_tolerance, "颜色容差", 8, 64)
        _integer(self.edge_padding, "描边范围", 0, 8)
        if self.glyph_height is not None:
            _integer(self.glyph_height, "字高", 8, 120)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SubtitleStyle":
        if not isinstance(data, dict):
            raise ValueError("字幕样式必须是对象")
        fields = {"hue", "saturation", "value", "color_tolerance", "glyph_height", "edge_padding"}
        if not {"hue", "saturation", "value"} <= data.keys() or data.keys() - fields:
            raise ValueError("字幕样式字段缺失或包含未知字段")
        return cls(**data)


def _validate_image(image: np.ndarray):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("需要非空的三通道图像")


def style_color_mask(hsv: np.ndarray, style: SubtitleStyle) -> np.ndarray:
    """Return colour candidates; low-contrast scenery is filtered by the caller.

    Brightness may fall at antialiased stroke edges. Hue wraps at 180, and is
    deliberately ignored for near-white strokes where hue is not meaningful.
    """
    _validate_image(hsv)
    if not isinstance(style, SubtitleStyle):
        raise ValueError("字幕样式无效")
    hue, saturation, value = cv2.split(hsv)
    tolerance = style.color_tolerance
    bright = value >= max(120, style.value - max(45, tolerance * 3))
    if style.saturation < 70:
        return bright & (saturation <= min(110, max(50, style.saturation + tolerance * 2)))
    hue_difference = np.abs(hue.astype(np.int16) - style.hue)
    hue_difference = np.minimum(hue_difference, 180 - hue_difference)
    return (bright & (hue_difference <= min(24, max(4, tolerance // 2))) &
            (saturation >= max(60, style.saturation - tolerance * 4)) &
            (saturation <= min(255, style.saturation + tolerance * 3)))


def _estimated_height(stats: np.ndarray, selected: np.ndarray) -> int | None:
    """Only infer height from several aligned, similarly sized glyph components.

    One disconnected Chinese stroke says little about font size. In that common
    case None lets the caller retain its usual conservative stroke filter.
    """
    x, y, width, height, area = [int(v) for v in selected]
    if not (12 <= height <= 80 and .35 * height <= width <= 1.8 * height and .08 <= area / (width * height) <= .72):
        return None
    peers = []
    for px, py, pw, ph, pa in stats[1:]:
        if (height * .8 <= ph <= height * 1.2 and .35 * ph <= pw <= 1.8 * ph and
                abs((py + ph / 2) - (y + height / 2)) <= height * .25 and
                abs((px + pw / 2) - (x + width / 2)) <= height * 5 and
                .08 <= pa / max(1, pw * ph) <= .72):
            peers.append(int(ph))
    return int(round(float(np.median(peers)))) if len(peers) >= 3 else None


def calibrate_style(image_bgr: np.ndarray, point_xy: tuple[int, int]) -> SubtitleStyle:
    """Calibrate from a clicked bright stroke or raise a helpful ValueError.

    Only the small clicked neighbourhood supplies colour. Large bright connected
    areas, dark clicks, and regions without local contrast are rejected rather
    than being learned as subtitle styling. This is not an OCR classifier.
    """
    _validate_image(image_bgr)
    if (not isinstance(point_xy, (tuple, list)) or len(point_xy) != 2 or
            any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in point_xy)):
        raise ValueError("请在字幕笔画内点击")
    x, y = (int(v) for v in point_xy)
    height, width = image_bgr.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError("点击位置不在图像内")
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    clicked = hsv[y, x]
    if int(clicked[2]) < 145:
        raise ValueError("这里较暗，请点击中文字幕内部的明亮笔画")
    # Prefer the bright interior in a 7px neighbourhood if the click hit an
    # antialiased edge. Never sample the full selected subtitle rectangle.
    patch = hsv[max(0, y - 3):min(height, y + 4), max(0, x - 3):min(width, x + 4)]
    preliminary = SubtitleStyle(*(int(v) for v in clicked))
    similar = patch[style_color_mask(patch, preliminary)]
    upper = similar[similar[:, 2] >= np.percentile(similar[:, 2], 65)]
    # Circular hue around the clicked colour avoids averaging red H=179 and H=0
    # to cyan. For white, hue has no effect but remains serializable.
    hue_delta = (upper[:, 0].astype(np.int16) - int(clicked[0]) + 90) % 180 - 90
    color = (int(round(int(clicked[0]) + float(np.median(hue_delta)))) % 180,
             int(round(float(np.median(upper[:, 1])))),
             int(round(float(np.median(upper[:, 2])))))
    style = SubtitleStyle(*color)
    candidates = style_color_mask(hsv, style).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, 8)
    label = int(labels[y, x])
    if not label:
        raise ValueError("该点靠近笔画边缘，请点击字幕笔画内部")
    left, top, component_width, component_height, area = [int(v) for v in stats[label]]
    if area < 3 or max(component_width, component_height) > 120 or area > 2500:
        raise ValueError("这里不像局部字幕笔画，请避开背景亮块重新点击")
    component = (labels[top:top + component_height, left:left + component_width] == label).astype(np.uint8)
    thickness = float(cv2.distanceTransform(np.pad(component, 1), cv2.DIST_L2, 3).max())
    density = area / (component_width * component_height)
    if thickness > 5.5 or (min(component_width, component_height) >= 8 and density > .84):
        raise ValueError("选中区域过厚，可能是背景亮块；请点击字幕的细笔画")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    margin = 8
    l, t = max(0, left - margin), max(0, top - margin)
    r, b = min(width, left + component_width + margin), min(height, top + component_height + margin)
    local_component = (labels[t:b, l:r] == label).astype(np.uint8)
    ring = cv2.dilate(local_component, np.ones((7, 7), np.uint8)) & (1 - local_component)
    surrounding = gray[t:b, l:r][ring > 0]
    foreground = gray[t:b, l:r][local_component > 0]
    if not surrounding.size or float(np.median(foreground)) - float(np.percentile(surrounding, 25)) < 18:
        raise ValueError("这里与背景对比不足，请点击带清晰边缘的明亮字幕笔画")
    return SubtitleStyle(*color, glyph_height=_estimated_height(stats, stats[label]))
