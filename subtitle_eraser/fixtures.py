"""Deterministic synthetic video fixtures, with known clean backgrounds and masks."""
from __future__ import annotations

from pathlib import Path
import math
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .pipeline import TextRegion

def fixture_font():
    override = os.environ.get("SUBTITLE_ERASER_TEST_FONT")
    if override:
        candidate = Path(override)
        if not candidate.is_file():
            raise FileNotFoundError("SUBTITLE_ERASER_TEST_FONT does not point to a font file")
        return candidate
    system = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc"
    bundled = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSansSC.ttf"
    for candidate in (system, bundled):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Chinese sample font missing; retain subtitle_eraser/assets/fonts with the project")


FONT = fixture_font()
SCENES = ("纯色背景", "复杂纹理", "人脸示意", "快速运动", "镜头切换", "字幕换行")


@lru_cache(maxsize=12)
def load_font(size):
    font = ImageFont.truetype(str(FONT), size)
    # The bundled variable Noto font defaults to Thin, unlike the Regular
    # Windows fixture font. Choose its named Regular instance explicitly.
    try:
        if b"Regular" in font.get_variation_names():
            font.set_variation_by_name(b"Regular")
    except OSError:  # Static fonts do not expose variable-font names.
        pass
    return font


def sample_frame(index: int, width: int = 960, height: int = 540, fps: int = 24):
    """Return BGR subtitle frame, clean frame, exact glyph mask, OCR reference boxes."""
    time = index / fps
    scene = int(time / 2) % 6
    y, x = np.mgrid[:height, :width].astype(np.float32)
    scale = width / 960
    base = np.empty((height, width, 3), np.uint8)
    if scene == 0:
        base[:] = (76, 53, 35)
    elif scene == 1:
        base[:, :, 0] = np.clip(70 + 30*np.sin(x/11+time) + 15*np.cos(y/7), 0, 255)
        base[:, :, 1] = np.clip(85 + 35*np.cos(y/14) + 12*np.sin(x/5), 0, 255)
        base[:, :, 2] = np.clip(60 + 25*np.sin((x+y)/18), 0, 255)
    elif scene == 2:
        base[:] = (90, 72, 53)
        # Draw a simple illustration, deliberately crossing the subtitle band.
        cv2.ellipse(base, (width//2, int(height*.70)), (int(110*scale), int(160*scale)), 0, 0, 360, (125, 177, 212), -1)
        for dx in (-40, 40):
            cv2.circle(base, (width//2+int(dx*scale), int(height*.62)), int(10*scale), (40, 42, 55), -1)
        cv2.ellipse(base, (width//2, int(height*.83)), (int(35*scale), int(10*scale)), 0, 0, 180, (50, 70, 115), 3)
    elif scene == 3:
        base[:] = (60, 70, 80)
        for i in range(8):
            px = int((i * 173*scale + time * 280*scale) % (width+150*scale)-150*scale)
            cv2.rectangle(base, (px, 0), (px+int(95*scale), height), (70+i*8, 110, 110+i*10), -1)
    else:
        base[:, :, 0] = (x / width * 75 + 35).astype(np.uint8)
        base[:, :, 1] = (y / height * 75 + 45).astype(np.uint8)
        base[:, :, 2] = 65 if scene != 4 or index % 24 < 12 else 145
    # A moving object supplies a known temporal signal in every scene.
    center = (int(width*(.5+.32*math.sin(time))), int(height*.30))
    cv2.circle(base, center, int(38*scale), (190, 160, 75), -1)
    cv2.putText(base, f"SYNTHETIC TEST  {scene+1}/6", (int(28*scale), int(45*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, .7*scale, (220, 225, 235), 1, cv2.LINE_AA)
    cv2.putText(base, SCENES[scene].encode('ascii', errors='ignore').decode() or f"SCENE {scene+1}",
                (int(28*scale), int(75*scale)), cv2.FONT_HERSHEY_SIMPLEX, .5*scale, (190, 200, 210), 1)

    clean = base.copy()
    image = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    glyph = Image.new("L", (width, height))
    glyph_draw = ImageDraw.Draw(glyph)
    regions = []
    font_cn = load_font(max(14, int(30*scale)))
    font_en = load_font(max(12, int(24*scale)))
    # Each scene includes a 0.4 second blank interval to verify mask disappearance.
    visible = index % (fps*2) < int(fps*1.6)
    if visible:
        chinese_lines = ["只隐藏中文，保留英文"] if scene != 5 else ["字幕换行时", "也要保留英文"]
        for i, text in enumerate(chinese_lines):
            top = int(height*.73) - (len(chinese_lines)-1-i)*int(36*scale)
            bbox = draw.textbbox((0, 0), text, font=font_cn, stroke_width=2)
            left = int((width-(bbox[2]-bbox[0]))/2)
            fill = (255, 225, 65) if scene == 1 else (248, 248, 248)
            draw.text((left, top), text, font=font_cn, fill=fill, stroke_width=2, stroke_fill=(12, 12, 12))
            glyph_draw.text((left, top), text, font=font_cn, fill=255, stroke_width=2, stroke_fill=255)
            b = draw.textbbox((left, top), text, font=font_cn, stroke_width=2)
            regions.append(TextRegion(np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],np.float32),text))
        text = "Keep the English. Enjoy the video."
        bbox = draw.textbbox((0, 0), text, font=font_en, stroke_width=2)
        left, top = int((width-(bbox[2]-bbox[0]))/2), int(height*.84)
        draw.text((left, top), text, font=font_en, fill=(248,248,248), stroke_width=2, stroke_fill=(12,12,12))
        b = draw.textbbox((left, top), text, font=font_en, stroke_width=2)
        regions.append(TextRegion(np.array([[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]],np.float32),text))
        # Ground truth keeps English, removing only Chinese.
        clean_img = Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))
        ImageDraw.Draw(clean_img).text((left, top), text, font=font_en, fill=(248,248,248), stroke_width=2, stroke_fill=(12,12,12))
        clean = cv2.cvtColor(np.asarray(clean_img), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), clean, np.asarray(glyph), regions
