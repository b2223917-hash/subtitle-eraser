"""One-shot image registration when a player changes layout; never per-frame OCR.

Only a supported scale/translation of the selected subtitle itself is accepted.
Background-only matches (camera motion), sidebar matches and uncertain results
must not move an opaque repair layer onto arbitrary video pixels.
"""
from __future__ import annotations

import cv2
import numpy as np


def _gray(image):
    scale = min(1.0, 1100 / image.shape[1])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray, np.array([gray.shape[1]/image.shape[1], gray.shape[0]/image.shape[0]])


def relocate_region(before, after, roi):
    """Return a normalized region or None, without modifying either input.

    Burned-in bilingual subtitles scale with video content, independently of
    sidebars/browser chrome. Match that content, not the outer window's ratios.
    A scene/subtitle cut can legitimately make automatic relocation impossible.
    """
    old, old_scale = _gray(before)
    new, new_scale = _gray(after)
    oh, ow = old.shape
    rx, ry, rw, rh = roi
    x, y, w, h = np.array([rx*ow, ry*oh, rw*ow, rh*oh])
    allowed = np.zeros_like(old)
    # Near the selected line, avoiding the player's unrelated sidebar/chrome.
    xa, xb = max(0, int(x)), min(ow, int(x+w+1))
    ya, yb = max(0, int(y-max(5*h, .3*oh))), min(oh, int(y+2*h+1))
    allowed[ya:yb, xa:xb] = 255
    sift = cv2.SIFT_create(nfeatures=2200, contrastThreshold=.025)
    a, da = sift.detectAndCompute(old, allowed)
    b, db = sift.detectAndCompute(new, None)
    if da is None or db is None or min(len(a), len(b)) < 12:
        return None
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    matches = [p for pair in pairs if len(pair) == 2 for p, q in [pair] if p.distance < .70*q.distance]
    if len(matches) < 12:
        return None
    src = np.float32([a[m.queryIdx].pt for m in matches])
    dst = np.float32([b[m.trainIdx].pt for m in matches])
    matrix, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=2.0, maxIters=2000, confidence=.995)
    if matrix is None or inliers is None:
        return None
    valid = inliers.ravel().astype(bool)
    if valid.sum() < 12 or valid.mean() < .60:
        return None
    # No perspective warp or substantial rotation is expected for fullscreen.
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    physical_scale = scale * old_scale[0] / new_scale[0]
    if not .3 <= physical_scale <= 4.5 or abs(matrix[1, 0]) > .008*scale:
        return None
    line = valid & (src[:, 1] >= y-2) & (src[:, 1] <= y+h+2)
    if line.sum() < 6 or np.ptp(src[line, 0]) < max(20, .18*w):
        return None
    # Verify the subtitle's pixels too: a background transform must not move a
    # line which is separately rendered at a fixed screen size/position.
    warped = cv2.warpAffine(new, matrix, (ow, oh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    ya, yb = max(0, int(y)), min(oh, int(y+h+1))
    old_edges = cv2.Canny(old[ya:yb, xa:xb], 70, 150)
    new_edges = cv2.Canny(warped[ya:yb, xa:xb], 70, 150)
    if np.count_nonzero(old_edges) < 30:
        return None
    nearby = cv2.dilate(new_edges, np.ones((3, 3), np.uint8))
    if np.count_nonzero(nearby[old_edges > 0]) / np.count_nonzero(old_edges) < .65:
        return None
    corners = np.float64([[x, y, 1], [x+w, y+h, 1]]) @ matrix.T
    corners /= new_scale
    nh, nw = after.shape[:2]
    left, top = corners[0]
    right, bottom = corners[1]
    if left < -1 or top < -1 or right > nw+1 or bottom > nh+1:
        return None
    left, top = max(0., left), max(0., top)
    right, bottom = min(float(nw), right), min(float(nh), bottom)
    if right-left < 20 or bottom-top < 12:
        return None
    return left/nw, top/nh, (right-left)/nw, (bottom-top)/nh
