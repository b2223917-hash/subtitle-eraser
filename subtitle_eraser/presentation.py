"""Compose a complete processed crop before publishing it to the overlay.

Glyph-only alpha exposes new source glyphs outside the previous mask until a
new result arrives. Opaque crop presentation isolates that time gap; it delays
the crop's background by the same amount. It does not improve segmentation.
"""
import cv2
import numpy as np


def overlay_rgba(repaired: np.ndarray, mask: np.ndarray, *, cover_region=False) -> np.ndarray:
    if repaired.shape[:2] != mask.shape or repaired.ndim != 3 or repaired.shape[2] != 3:
        raise ValueError("Overlay image and mask dimensions must match")
    rgba = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = 255 if cover_region else mask
    return np.ascontiguousarray(rgba)
