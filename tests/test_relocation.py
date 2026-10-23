from time import perf_counter
import cv2
import numpy as np
import pytest
from subtitle_eraser.relocation import relocate_region
from subtitle_eraser.worker import ProcessingWorker, Packet
from subtitle_eraser.pipeline import Settings
from subtitle_eraser.windows import WindowInfo
from subtitle_eraser.style import SubtitleStyle


from subtitle_eraser.diagnostics import transition_fixture as layouts


def test_fullscreen_maps_video_not_sidebar_and_keeps_english_outside():
    before, after, roi, expected = layouts()
    found = relocate_region(before, after, roi)
    assert found is not None
    assert np.max(np.abs(np.array(found)-expected)*[1000, 600, 1000, 600]) < 2
    assert (found[1]+found[3])*600 < 540  # English starts lower down.
    restored = relocate_region(after, before, found)
    assert np.max(np.abs(np.array(restored)-roi)*[740, 380, 740, 380]) < 2


def test_unrelated_cut_and_background_only_matches_do_not_move_overlay():
    before, after, roi, _ = layouts()
    assert relocate_region(before, np.zeros_like(after), roi) is None
    after[450:530] = 40  # Background still matches; selected text no longer does.
    assert relocate_region(before, after, roi) is None


def test_worker_resize_recovers_and_remembers_previous_layout():
    before, after, roi, expected = layouts()
    worker = ProcessingWorker(demo=True)
    worker.latest_frame = before, perf_counter()
    worker.configure(roi, Settings(mode='manual'))
    worker._accept_frame(before, perf_counter())
    epoch = worker.epoch
    found, _, new_epoch = worker._accept_frame(after, perf_counter())
    assert found is not None and new_epoch > epoch
    assert not worker.publish_packet(Packet(1, None, 0, 0, (740, 380), (0, 0, 1, 1), epoch))
    # Returning after a scene cut uses the layout learned in this connection.
    restored, _, _ = worker._accept_frame(np.zeros_like(before), perf_counter())
    assert restored == roi


def test_retarget_same_sized_window_invalidates_and_relocates():
    before, after, roi, _ = layouts()
    worker = ProcessingWorker(window=WindowInfo(123, 'original'))
    worker.latest_frame = before, 0
    worker.configure(roi, Settings(mode='manual'))
    worker._accept_frame(before, 0)
    worker.request_window(WindowInfo(456, 'fullscreen'))
    assert worker._accept_frame(before, 1)[0] is None  # Old-source frame must not publish.
    worker._requested_window = None
    worker.window = WindowInfo(456, 'fullscreen')
    worker._force_relocation = True
    assert worker._accept_frame(after, 2)[0] is not None


def test_explicit_reselection_during_registration_wins(monkeypatch):
    before, after, roi, _ = layouts()
    worker = ProcessingWorker(demo=True)
    worker.latest_frame = before, 0
    worker.configure(roi, Settings(mode='manual'))
    selected = (.2, .6, .5, .1)
    def interrupted(*_):
        worker.configure(selected, Settings(mode='manual'))
        return roi
    monkeypatch.setattr('subtitle_eraser.worker.relocate_region', interrupted)
    assert worker._accept_frame(after, 1)[0] == selected


def test_calibrated_glyph_height_scales_with_video_and_back():
    before, after, roi, _ = layouts()
    worker = ProcessingWorker(demo=True)
    worker.latest_frame = before, 0
    settings = Settings(mode='manual', style=SubtitleStyle(0, 0, 255, glyph_height=18))
    worker.configure(roi, settings)
    worker._accept_frame(before, 0)
    found, scaled, _ = worker._accept_frame(after, 1)
    assert scaled.style.glyph_height == 36
    assert worker.configuration_snapshot()[2].style.glyph_height == 36
    assert worker.settings == settings  # No accumulated rounding on repeated transitions.
    assert worker._accept_frame(before, 2)[1].style.glyph_height == 18


def test_window_request_during_registration_discards_current_source(monkeypatch):
    before, after, roi, _ = layouts()
    worker = ProcessingWorker(window=WindowInfo(123, 'original'))
    worker.latest_frame = before, 0
    worker.configure(roi, Settings(mode='manual'))
    def interrupted(*_):
        worker.request_window(WindowInfo(456, 'other fullscreen'))
        return roi
    monkeypatch.setattr('subtitle_eraser.worker.relocate_region', interrupted)
    assert worker._accept_frame(after, 1)[0] is None
    assert worker.snapshot()[1] is None
