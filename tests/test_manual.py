"""Low-latency mode must work immediately without a language-recognition model."""
import cv2
import numpy as np
import pytest

from subtitle_eraser.fixtures import sample_frame
from subtitle_eraser.pipeline import Settings, SubtitleProcessor, polygon_mask


def chinese_crop(image):
    height, width = image.shape[:2]
    return (slice(int(height * .65), int(height * .82)),
            slice(int(width * .10), int(width * .90)))


@pytest.mark.parametrize("scene", range(6))
def test_manual_first_frame_without_ocr_protects_english_and_unmasked_pixels(monkeypatch, scene):
    def forbidden_ocr(*args, **kwargs):
        raise AssertionError("Manual mode must not create OCR")

    monkeypatch.setattr("subtitle_eraser.pipeline.OcrDetector", forbidden_ocr)
    image, clean, truth, regions = sample_frame(scene * 48)
    roi = chinese_crop(image)
    processor = SubtitleProcessor(settings=Settings(mode="manual"))
    result = processor.process(image[roi], force_detection=True)
    assert processor.detector is None
    assert result.timing["ocr_ms"] == 0
    assert np.any(result.mask), "The first frame must produce a mask without warmup"
    assert np.count_nonzero(truth[roi]) == np.count_nonzero(truth)
    recall = np.count_nonzero((result.mask > 0) & (truth[roi] > 0)) / np.count_nonzero(truth)
    assert recall > .95
    assert np.array_equal(result.original[result.mask == 0], result.repaired[result.mask == 0])
    composed = image.copy()
    composed[roi] = result.repaired
    english = polygon_mask(image.shape[:2], [r for r in regions if not r.chinese], 3) > 0
    assert np.array_equal(image[english], composed[english])
    before = np.abs(image[truth > 0].astype(float) - clean[truth > 0]).mean()
    after = np.abs(composed[truth > 0].astype(float) - clean[truth > 0]).mean()
    assert after < before


def test_manual_clears_subtitles_and_handles_cut_and_wrapping_without_stale_pixels():
    class ForbiddenDetector:
        def detect(self, image):
            raise AssertionError("Manual mode must not call an injected OCR detector")

    processor = SubtitleProcessor(ForbiddenDetector(), Settings(mode="manual"))
    # The sequence contains every scene, hard scene changes, and each blank interval.
    for index in [i for scene in range(6) for i in (scene * 48, scene * 48 + 38, scene * 48 + 47)]:
        image, _, truth, _ = sample_frame(index)
        roi = chinese_crop(image)
        result = processor.process(image[roi])
        assert result.timing["ocr_ms"] == 0
        assert result.timing["background_ocr_ms"] == 0
        if np.any(truth):
            assert np.any(result.mask)
        else:
            assert not np.any(result.mask), f"Unexpected background repair on blank frame {index}"
            assert np.array_equal(image[roi], result.repaired)


@pytest.mark.parametrize("size", [(960, 540), (1440, 810)])
def test_manual_auto_color_does_not_mistake_illustrated_skin_for_yellow_text(size):
    processor = SubtitleProcessor(settings=Settings(mode="manual", font_color="auto"))
    for index in (96, 114, 133):
        image, _, truth, _ = sample_frame(index, *size)
        roi = chinese_crop(image)
        result = processor.process(image[roi])
        allowed_edge = cv2.dilate(truth[roi], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        # Glyph dilation may cover three extra pixels, but skin farther away is
        # scenery. This fixture previously lost thousands of those pixels.
        assert not np.any((result.mask > 0) & (allowed_edge == 0))
        assert np.count_nonzero((result.mask > 0) & (truth[roi] > 0)) / np.count_nonzero(truth) > .95
