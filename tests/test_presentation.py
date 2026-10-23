import cv2
import numpy as np
import pytest

from subtitle_eraser.presentation import overlay_rgba
from tools.transition_benchmark import (DELAY_MS, HEIGHT, ROI, SCENARIOS,
                                        STEP_MS, SWITCH_MS, WIDTH, compose,
                                        crop, simulate_frame)


@pytest.mark.parametrize("empty", [False, True])
def test_full_crop_keeps_repaired_pixels_opaque_even_when_subtitles_disappear(empty):
    repaired = np.arange(8 * 11 * 3, dtype=np.uint8).reshape(8, 11, 3)
    mask = np.zeros((8, 11), np.uint8)
    if not empty:
        mask[2:5, 3:8] = 255
    before_image, before_mask = repaired.copy(), mask.copy()
    result = overlay_rgba(repaired, mask, cover_region=True)
    assert np.all(result[:, :, 3] == 255)
    assert np.array_equal(result[:, :, :3], cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB))
    assert np.array_equal(repaired, before_image)
    assert np.array_equal(mask, before_mask)
    assert result.flags.c_contiguous


def test_glyph_only_mode_preserves_mask_alpha():
    image = np.full((4, 7, 3), (11, 24, 36), np.uint8)
    mask = np.zeros((4, 7), np.uint8)
    mask[1:3, 2:5] = 255
    rgba = overlay_rgba(image, mask)
    assert np.array_equal(rgba[:, :, 3], mask)
    assert np.all(rgba[:, :, :3] == (36, 24, 11))


def test_overlay_composites_processed_pixels_and_preserves_everything_outside_crop():
    source = np.full((HEIGHT, WIDTH, 3), (19, 41, 91), np.uint8)
    # Distinct pixels ensure accidentally using the raw source fails this test.
    processed = np.full(crop(source).shape, (72, 52, 32), np.uint8)
    rgba = overlay_rgba(processed, np.zeros(processed.shape[:2], np.uint8), cover_region=True)
    result = compose(source, rgba)
    assert np.array_equal(crop(result), processed)
    outside = np.ones((HEIGHT, WIDTH), bool)
    crop(outside)[:] = False
    assert np.array_equal(result[outside], source[outside])
    assert np.array_equal(result[200:244], source[200:244])


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_transition_does_not_expose_new_source_glyphs_during_processing_delay(scenario):
    samples = []
    for time_ms in range(SWITCH_MS, SWITCH_MS + DELAY_MS + STEP_MS, STEP_MS):
        raw, glyph_only, continuous, row = simulate_frame(scenario, time_ms)
        samples.append(row)
        assert row["opaque_crop_exposed_source_pixels"] == 0
        assert row["opaque_crop_min_alpha"] == 255
        assert np.array_equal(raw[200:244], continuous[200:244])
    assert all(s["glyph_only_exposed_source_pixels"] > 0 for s in samples[:-1])
    assert samples[-1]["glyph_only_exposed_source_pixels"] == 0
    if scenario == "blank_to_new":
        assert all(s["prior_mask_empty"] for s in samples[:-1])


def test_incompatible_image_and_mask_fail_before_compositing():
    with pytest.raises(ValueError, match="dimensions"):
        overlay_rgba(np.zeros((5, 10, 3), np.uint8), np.zeros((6, 10), np.uint8), cover_region=True)
