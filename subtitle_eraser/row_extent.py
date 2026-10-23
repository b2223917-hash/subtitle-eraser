"""Extend a manual Chinese band horizontally through an aligned text cluster.

Never moves the user's vertical language boundary or bridges a distant object.
No OCR, historical pixels, or whole-screen repair is involved.
"""
import cv2
import numpy as np
from dataclasses import replace
from .pipeline import stroke_candidates


def fit_manual_row(image, roi, settings):
    if settings.mode != "manual":
        return roi
    ih, iw = image.shape[:2]
    rx, ry, rw, rh = roi
    top, bottom = max(0, round(ry*ih)), min(ih, round((ry+rh)*ih))
    if bottom-top < 12:
        return roi
    ratio = min(1., 64/(bottom-top), 1600/iw)
    band = image[top:bottom]
    if ratio < 1:
        band = cv2.resize(band, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    scale_x = band.shape[1]/iw
    style = settings.style
    if style is not None and style.glyph_height is not None:
        style = replace(style, glyph_height=max(8, round(style.glyph_height*ratio)))
    seeds, _ = stroke_candidates(band, replace(settings, style=style))
    _, _, stats, _ = cv2.connectedComponentsWithStats(seeds, 8)
    hband = band.shape[0]
    parts = []
    for x, y, w, h, area in stats[1:]:
        if (max(4, hband*.18) <= h <= hband*.98 and 1 <= w <= hband*1.6 and
                area >= 4 and .04 <= area/max(1, w*h) < .86):
            parts.append((int(x), int(y), int(w), int(h)))
    left, right = rx*iw*scale_x, (rx+rw)*iw*scale_x
    anchors = [p for p in parts if p[0]+p[2] > left and p[0] < right]
    if len(anchors) < 3:
        return roi
    typical_height = float(np.percentile([p[3] for p in anchors], 75))
    tall = [p for p in anchors if p[3] >= typical_height*.6]
    center = float(np.median([p[1]+p[3]/2 for p in tall]))
    aligned = sorted(p for p in parts if p[3] >= typical_height*.3 and
                     p[3] <= typical_height*1.5 and abs(p[1]+p[3]/2-center) <= typical_height*.35)
    groups = []
    for p in aligned:
        if not groups or p[0]-max(q[0]+q[2] for q in groups[-1]) > typical_height*1.2:
            groups.append([])
        groups[-1].append(p)
    pad = max(2., (settings.edge_padding+1)*settings.processing_scale*scale_x, typical_height*.13)
    for group in groups:
        if sum(p in anchors for p in group) < 3:
            continue
        lo, hi = min(p[0] for p in group), max(p[0]+p[2] for p in group)
        # Chinese side strokes may be disconnected, short horizontal bars. They
        # cannot establish a text row, but can extend a row already supported by
        # several tall glyph components. Avoid cutting the last character's arm.
        small = [(x, w) for x, y, w, h, area in stats[1:] if area >= 2 and
                 w <= typical_height*1.6 and h <= typical_height*1.2 and
                 x+w >= lo-typical_height*.35 and x <= hi+typical_height*.35 and
                 center-typical_height*.7 <= y+h/2 <= center+typical_height*.7]
        if small:
            lo, hi = min(lo, min(p[0] for p in small)), max(hi, max(p[0]+p[1] for p in small))
        left, right = min(left, lo-pad), max(right, hi+pad)
    left, right = max(0., left/scale_x), min(float(iw), right/scale_x)
    # Keep the band stable: grow only, and preserve its exact normalized Y range.
    return (left/iw, ry, (right-left)/iw, rh) if left < rx*iw-1 or right > (rx+rw)*iw+1 else roi
