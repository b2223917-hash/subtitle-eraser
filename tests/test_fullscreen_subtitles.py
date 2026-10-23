"""Real Chinese text, scaled outlines, changing line length, and selection focus."""
from time import perf_counter
import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication
from subtitle_eraser.fixtures import sample_frame
from subtitle_eraser.pipeline import Settings, SubtitleProcessor, polygon_mask
from subtitle_eraser.row_extent import fit_manual_row
from subtitle_eraser.worker import ProcessingWorker
from subtitle_eraser.ui import SelectionDialog


@pytest.mark.parametrize('scale', [1.5, 2, 3])
@pytest.mark.parametrize('fresh_fullscreen', [False, True])
def test_scaled_chinese_outline_and_shadow_are_covered(scale, fresh_fullscreen):
    image, clean, truth, regions = sample_frame(0)
    crop = np.s_[351:443, 96:864]
    source = cv2.resize(image[crop], None, fx=scale, fy=scale)
    expected = cv2.resize(clean[crop], None, fx=scale, fy=scale)
    glyphs = cv2.resize(truth[crop], None, fx=scale, fy=scale) > 8
    outlines = glyphs & (cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) < 60)
    settings = Settings(mode='manual', processing_scale=1 if fresh_fullscreen else scale)
    result = SubtitleProcessor(settings=settings).process(source)
    assert result.safe_to_present
    assert np.mean(result.mask[glyphs] > 0) > .995
    assert np.mean(result.mask[outlines] > 0) > .995
    assert np.abs(result.repaired[glyphs].astype(float)-expected[glyphs]).mean() < 3
    assert np.array_equal(source[result.mask == 0], result.repaired[result.mask == 0])
    assert result.timing['ocr_ms'] == 0


@pytest.mark.parametrize('scale', [1, 2, 3])
def test_row_fits_both_missing_ends_without_moving_english_boundary(scale):
    image, _, truth, regions = sample_frame(0)
    x, y, w, h = cv2.boundingRect(truth)
    roi = ((x+w*.22)/960, (y-4)/540, w*.56/960, (h+8)/540)
    source = cv2.resize(image, None, fx=scale, fy=scale)
    settings = Settings(mode='manual', processing_scale=scale)
    fitted = fit_manual_row(source, roi, settings)
    assert fitted[1] == roi[1] and fitted[3] == roi[3]
    ih, iw = source.shape[:2]
    left, top, width, height = np.round(np.array(fitted)*[iw, ih, iw, ih]).astype(int)
    selected = np.zeros((ih, iw), bool)
    selected[top:top+height, left:left+width] = True
    full_truth = cv2.resize(truth, (iw, ih)) > 8
    assert np.all(selected[full_truth])
    result = SubtitleProcessor(settings=settings).process(source[top:top+height, left:left+width])
    composed = source.copy()
    composed[top:top+height, left:left+width] = result.repaired
    english = cv2.resize(polygon_mask(image.shape[:2], [r for r in regions if not r.chinese], 3), (iw, ih)) > 0
    assert np.array_equal(source[english], composed[english])


def test_row_does_not_grow_toward_distant_bright_background():
    image, _, truth, _ = sample_frame(0)
    x, y, w, h = cv2.boundingRect(truth)
    roi = ((x-8)/960, (y-4)/540, (w+16)/960, (h+8)/540)
    cv2.rectangle(image, (20, y), (60, y+h), (250, 250, 250), -1)
    cv2.putText(image, '123', (810, y+h-4), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    assert fit_manual_row(image, roi, Settings(mode='manual')) == roi


def test_selection_is_remembered_before_any_processed_frame():
    worker = ProcessingWorker(demo=True)
    full = np.zeros((1080, 1920, 3), np.uint8)
    small = np.zeros((540, 960, 3), np.uint8)
    worker.latest_frame = full, perf_counter()
    roi = (.2, .7, .6, .1)
    worker.configure(roi, Settings(mode='manual'))
    worker._accept_frame(small, perf_counter())  # Focus changes layout before processing.
    assert worker._accept_frame(full, perf_counter())[0] == roi


def test_saved_fullscreen_selection_survives_focus_resize_and_scene_cut():
    worker = ProcessingWorker(demo=True)
    full = np.zeros((1080, 1920, 3), np.uint8)
    small = np.zeros((540, 960, 3), np.uint8)
    worker._accept_frame(full, 0)
    frame, key = worker.selection_snapshot()
    worker._accept_frame(small, 1)  # Opening the selector exits fullscreen.
    roi = (.2, .7, .6, .06)
    assert not worker.configure_selection(roi, Settings(mode='manual'), frame[0], key)
    assert worker.roi is None
    new_scene = np.full_like(full, 70)
    assert worker._accept_frame(new_scene, 2)[0] == roi


def test_4k_thin_selection_validity_uses_source_pixels():
    app = QApplication.instance() or QApplication([])
    dialog = SelectionDialog(np.zeros((2160, 3840, 3), np.uint8), None, manual=True)
    try:
        dialog.canvas.set_normalized_region((.2, .75, .6, 30/2160))
        assert dialog.canvas.selected.height() < 12
        assert dialog.ok.isEnabled()
    finally:
        dialog.close()
