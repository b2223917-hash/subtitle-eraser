"""Exercise calibrated manual processing on rendered Chinese/English fixtures."""
import cv2
import numpy as np
import pytest

from subtitle_eraser.fixtures import sample_frame
from subtitle_eraser.pipeline import Settings, SubtitleProcessor, polygon_mask
from subtitle_eraser.style import calibrate_style


def chinese_crop(width, height):
    return np.s_[int(height * .65):int(height * .82), int(width * .1):int(width * .9)]


def calibrate_fixture(image, truth):
    # Synthetic ground truth substitutes only for the user's stroke click. It
    # must never be passed to the processing/masking algorithm.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    interior = ((truth > 0) & (hsv[:, :, 2] >= 230)).astype(np.uint8)
    distance = cv2.distanceTransform(interior, cv2.DIST_L2, 3)
    y, x = np.unravel_index(np.argmax(distance), distance.shape)
    return calibrate_style(image, (int(x), int(y)))


@pytest.mark.parametrize("width,height", [(960, 540), (1440, 810)])
@pytest.mark.parametrize("scene", range(6))
def test_calibrated_manual_scenes_preserve_english_background_and_blank_frames(monkeypatch, width, height, scene):
    def forbidden_ocr(*args, **kwargs):
        raise AssertionError("Calibrated manual processing must not load OCR")

    monkeypatch.setattr("subtitle_eraser.pipeline.OcrDetector", forbidden_ocr)
    initial, _, initial_truth, _ = sample_frame(scene * 48, width, height)
    roi = chinese_crop(width, height)
    style = calibrate_fixture(initial[roi], initial_truth[roi])
    processor = SubtitleProcessor(settings=Settings(mode="manual", style=style))
    default = SubtitleProcessor(settings=Settings(mode="manual"))
    for offset in (0, 17, 37, 38, 47):
        image, clean, truth, regions = sample_frame(scene * 48 + offset, width, height)
        result = processor.process(image[roi], force_detection=True)
        baseline = default.process(image[roi])
        assert processor.detector is None
        assert result.timing["ocr_ms"] == result.timing["background_ocr_ms"] == 0
        assert np.count_nonzero(truth[roi]) == np.count_nonzero(truth)
        assert np.array_equal(result.repaired[result.mask == 0], image[roi][result.mask == 0])
        composed = image.copy()
        composed[roi] = result.repaired
        english = polygon_mask(image.shape[:2], [r for r in regions if not r.chinese], 3) > 0
        assert np.array_equal(composed[english], image[english])
        near_glyph = cv2.dilate(truth[roi], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        assert not np.any((result.mask > 0) & (near_glyph == 0)), "Calibrated style must not select surrounding scenery"
        if not np.any(truth):
            assert not np.any(result.mask)
            assert np.array_equal(result.repaired, image[roi])
            continue
        true_pixels = truth[roi] > 0
        recall = np.count_nonzero((result.mask > 0) & true_pixels) / np.count_nonzero(true_pixels)
        original_recall = np.count_nonzero((baseline.mask > 0) & true_pixels) / np.count_nonzero(true_pixels)
        assert recall > .98
        assert recall >= original_recall - .01
        before = np.abs(image[truth > 0].astype(float) - clean[truth > 0]).mean()
        after = np.abs(composed[truth > 0].astype(float) - clean[truth > 0]).mean()
        assert after < before


def test_calibration_is_reused_without_ocr_across_hard_cuts_and_text_wraps():
    class ForbiddenDetector:
        def detect(self, image):
            raise AssertionError("No OCR allowed for manual style processing")

    initial, _, truth, _ = sample_frame(0)
    roi = chinese_crop(960, 540)
    style = calibrate_fixture(initial[roi], truth[roi])
    processor = SubtitleProcessor(ForbiddenDetector(), Settings(mode="manual", style=style))
    # Same white style across face, motion, hard cut, wrapped text, disappearance.
    for index in (0, 96, 144, 192, 204, 240, 278):
        image, _, truth, _ = sample_frame(index)
        result = processor.process(image[roi])
        if np.any(truth):
            recall = np.count_nonzero((result.mask > 0) & (truth[roi] > 0)) / np.count_nonzero(truth)
            assert recall > .98
        else:
            assert not np.any(result.mask)
        assert result.timing["ocr_ms"] == 0
