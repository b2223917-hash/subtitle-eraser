import json

import cv2
import numpy as np
import pytest

from subtitle_eraser.style import SubtitleStyle, calibrate_style, style_color_mask


def stroke_frame(color=(245, 245, 245)):
    image = np.full((80, 180, 3), 25, np.uint8)
    # Crossed thin strokes have real local contrast, unlike a bright rectangle.
    cv2.line(image, (40, 20), (40, 44), color, 2)
    cv2.line(image, (30, 32), (50, 32), color, 2)
    return image


def test_style_round_trip_and_no_pixels_persisted():
    style = calibrate_style(stroke_frame(), (40, 25))
    payload = json.loads(json.dumps(style.to_dict()))
    assert SubtitleStyle.from_dict(payload) == style
    assert set(payload) == {"hue", "saturation", "value", "color_tolerance", "glyph_height", "edge_padding"}
    assert style.glyph_height is None  # One component cannot establish font size.


@pytest.mark.parametrize("patch", [
    {"hue": True}, {"hue": 180}, {"value": 100}, {"glyph_height": 500},
    {"color_tolerance": float("nan")}, {"edge_padding": -1}, {"unknown": 2},
])
def test_invalid_saved_styles_are_rejected(patch):
    data = SubtitleStyle(0, 0, 245).to_dict()
    data.update(patch)
    with pytest.raises(ValueError):
        SubtitleStyle.from_dict(data)


@pytest.mark.parametrize("data", [None, [], {}, {"hue": 0, "value": 245}])
def test_invalid_saved_payloads_are_rejected(data):
    with pytest.raises(ValueError):
        SubtitleStyle.from_dict(data)


def test_calibration_ignores_unrelated_bright_clothing():
    source = stroke_frame((45, 225, 255))
    expected = calibrate_style(source, (40, 25))
    source[:, 90:] = (255, 255, 255)
    actual = calibrate_style(source, (40, 25))
    assert actual == expected
    mask = style_color_mask(cv2.cvtColor(source, cv2.COLOR_BGR2HSV), actual)
    assert mask[25, 40]
    assert not mask[:, 100:].any()


@pytest.mark.parametrize("point", [(5, 5), (-1, 20), (500, 20), (True, 20), (2.5, 3)])
def test_dark_invalid_or_out_of_bounds_clicks_are_rejected(point):
    with pytest.raises(ValueError):
        calibrate_style(stroke_frame(), point)


def test_bright_background_block_is_not_learned_as_subtitles():
    source = stroke_frame()
    source[5:75, 85:175] = 250
    with pytest.raises(ValueError, match="亮块"):
        calibrate_style(source, (100, 30))
    with pytest.raises(ValueError):
        calibrate_style(np.full((40, 70, 3), 250, np.uint8), (20, 20))


def test_white_style_includes_antialiasing_but_excludes_dark_and_saturated_color():
    hsv = np.array([[[0, 0, 245], [100, 25, 180], [0, 0, 100], [20, 200, 245]]], np.uint8)
    assert style_color_mask(hsv, SubtitleStyle(0, 0, 245)).tolist() == [[True, True, False, False]]


def test_color_matching_wraps_red_hue():
    hsv = np.array([[[179, 240, 245], [1, 230, 200], [90, 240, 245], [0, 0, 245]]], np.uint8)
    assert style_color_mask(hsv, SubtitleStyle(179, 240, 245)).tolist() == [[True, True, False, False]]


def test_height_requires_multiple_similar_aligned_stroke_components():
    image = stroke_frame()
    for offset in (35, 70):
        cv2.line(image, (40 + offset, 20), (40 + offset, 44), (245, 245, 245), 2)
        cv2.line(image, (30 + offset, 32), (50 + offset, 32), (245, 245, 245), 2)
    style = calibrate_style(image, (40, 25))
    assert 24 <= style.glyph_height <= 28


def test_single_disconnected_line_does_not_invent_font_height():
    image = np.full((60, 160, 3), 20, np.uint8)
    cv2.line(image, (30, 30), (60, 30), (245, 245, 245), 2)
    assert calibrate_style(image, (40, 30)).glyph_height is None


def test_bright_thin_low_contrast_pattern_is_rejected():
    image = np.full((60, 160, 3), 235, np.uint8)
    cv2.line(image, (30, 30), (60, 30), (245, 245, 245), 2)
    with pytest.raises(ValueError):
        calibrate_style(image, (40, 30))


@pytest.mark.parametrize("image", [np.zeros((20, 30), np.uint8), np.zeros((0, 0, 3), np.uint8), np.zeros((20, 30, 3), float)])
def test_invalid_images_fail_clearly(image):
    with pytest.raises(ValueError):
        calibrate_style(image, (0, 0))
    with pytest.raises(ValueError):
        style_color_mask(image, SubtitleStyle(0, 0, 245))
