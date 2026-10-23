"""Separate controls/comparison behavior, using only synthetic in-memory data."""
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from subtitle_eraser import ui
from subtitle_eraser.pipeline import FrameResult
from subtitle_eraser.profiles import ProfileStore
from subtitle_eraser.worker import Packet


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    for name in ("register_toggle", "register_selection", "register_selection_fallback",
                 "exclude_from_capture"):
        monkeypatch.setattr(ui.windows, name, lambda *args: True)
    for name in ("unregister_toggle", "unregister_selection"):
        monkeypatch.setattr(ui.windows, name, lambda *args: None)
    monkeypatch.setattr(ui.windows, "list_windows", lambda: [])
    window = ui.MainWindow(ProfileStore(tmp_path / "profiles.json"))
    window.timer.stop()
    window.setStyleSheet(ui.STYLE)
    window.comparison.setStyleSheet(ui.STYLE)
    for widget in (window, window.comparison):
        widget.setAttribute(Qt.WA_DontShowOnScreen)
    try:
        yield window
    finally:
        window.worker = None
        window.close()
        window.deleteLater()
        app.sendPostedEvents()


def attach_packet(window, shade=60):
    frame = np.full((180, 320, 3), shade, np.uint8)
    crop = frame[120:160].copy()
    result = FrameResult(crop, np.zeros(crop.shape[:2], np.uint8), crop.copy(), [],
                         "测试画面", {"total_ms": 2.0})
    packet = Packet(1, result, perf_counter(), perf_counter(), (320, 180), (0, 120, 320, 40), 0)
    worker = SimpleNamespace(epoch=0, roi=(0, .66, 1, .22), source=None,
                             snapshot=lambda: ((frame, packet.captured_at), packet),
                             isRunning=lambda: True)
    window.worker = worker
    return worker, packet


def test_main_controls_fit_small_window_and_advanced_options_scroll(window):
    app = QApplication.instance()
    window.resize(640, 550)
    window.show()
    app.processEvents()
    assert window.width() == 640
    assert not window.advanced_panel.isVisible()
    assert not window.comparison.isVisible()
    assert window.centralWidget().rect().contains(window.hotkey_label.geometry())
    assert window.controls_scroll.horizontalScrollBar().maximum() == 0
    window.advanced_toggle.click()
    app.processEvents()
    assert window.advanced_panel.isVisible()
    assert window.controls_scroll.verticalScrollBar().maximum() > 0
    assert window.controls_scroll.horizontalScrollBar().maximum() == 0
    assert window.height() == 550


def test_initial_size_fits_a_scaled_1080p_work_area(window, monkeypatch):
    from PySide6.QtCore import QRect
    screen = SimpleNamespace(availableGeometry=lambda: QRect(0, 0, 1280, 680))
    monkeypatch.setattr(window, "screen", lambda: screen)
    ui.fit_window(window, (820, 730), (640, 480))
    assert window.height() <= 624
    assert window.width() <= 1280


def test_comparison_opens_current_static_result_and_closes_without_stopping_worker(window):
    worker, packet = attach_packet(window)
    window.last_sequence = packet.sequence  # This result already reached the overlay.
    window.open_comparison()
    QApplication.processEvents()
    assert all(preview.image is not None for preview in window.previews)
    assert window.comparison.cached_packet is packet
    window.comparison.close()
    assert not window.comparison.isVisible()
    assert window.worker is worker
    assert worker.isRunning()


def test_hidden_comparison_does_no_image_or_debug_conversion(window, monkeypatch):
    worker, packet = attach_packet(window)
    monkeypatch.setattr(packet.result, "debug_image", lambda: pytest.fail("Hidden debug rendering"))
    window.poll()
    assert window.packet is packet  # Processing/presentation keeps consuming results.
    assert all(preview.image is None for preview in window.previews)


def test_visible_comparison_updates_when_main_controls_are_minimized(window):
    worker, packet = attach_packet(window)
    window.showMinimized()
    window.open_comparison()
    QApplication.processEvents()
    changed = replace(packet, sequence=2,
                      result=replace(packet.result, original=np.full_like(packet.result.original, 200)))
    worker.snapshot = lambda: ((changed.result.original, changed.captured_at), changed)
    window.last_preview_at = 0
    window.poll()
    assert window.comparison.cached_packet is changed
    assert window.previews[0].image.pixelColor(0, 0).red() == 200


def test_freeze_preserves_selected_frame_when_switching_views(window):
    worker, packet = attach_packet(window)
    window.open_comparison()
    QApplication.processEvents()
    window.comparison.freeze_button.click()
    changed = replace(packet, sequence=2,
                      result=replace(packet.result, original=np.full_like(packet.result.original, 200)))
    worker.snapshot = lambda: ((changed.result.original, changed.captured_at), changed)
    window.last_preview_at = 0
    window.poll()
    assert window.packet is changed  # Freeze must never pause processing.
    assert window.comparison.cached_packet is packet
    window.comparison.view_tabs.setCurrentIndex(1)
    assert window.previews[0].image.pixelColor(0, 0).red() == 60
    window.comparison.freeze_button.click()
    assert window.comparison.cached_packet is changed
    assert window.previews[0].image.pixelColor(0, 0).red() == 200


def test_all_three_previews_are_complete_at_small_size_and_single_view_is_larger(window):
    attach_packet(window)
    window.comparison.resize(680, 500)
    window.open_comparison()
    app = QApplication.instance()
    app.processEvents()
    comparison = window.comparison
    for panel, preview, caption in zip(comparison.panels, window.previews, comparison.captions):
        assert panel.isVisible() and preview.isVisible()
        assert panel.rect().contains(preview.geometry())
        assert panel.rect().contains(caption.geometry())
        assert not preview.geometry().intersects(caption.geometry())
        viewport_rect = preview.rect().translated(preview.mapTo(comparison.scroll.viewport(), preview.rect().topLeft()))
        assert comparison.scroll.viewport().rect().contains(viewport_rect)
        image_rect = preview.image_rect()
        assert preview.rect().contains(image_rect.toAlignedRect())
        assert abs(image_rect.width()/image_rect.height()-8) < .4
    width = window.previews[2].width()
    comparison.view_tabs.setCurrentIndex(3)
    app.processEvents()
    assert window.previews[2].width() > width*2
    assert not comparison.panels[0].isVisible()
    comparison.view_tabs.setCurrentIndex(0)
    comparison.orientation.setCurrentIndex(1)
    app.processEvents()
    assert comparison.panels[1].y() > comparison.panels[0].y()
    assert comparison.panels[2].y() > comparison.panels[1].y()
    comparison.resize(680, 700)
    app.processEvents()
    for preview in window.previews:
        rect = preview.rect().translated(preview.mapTo(comparison.scroll.viewport(), preview.rect().topLeft()))
        assert comparison.scroll.viewport().rect().contains(rect)


def test_frozen_snapshot_exports_the_frozen_frame(window, monkeypatch, tmp_path):
    worker, packet = attach_packet(window)
    window.open_comparison()
    QApplication.processEvents()
    window.comparison.freeze_button.click()
    window.packet = replace(packet, sequence=3,
                            result=replace(packet.result, original=np.full_like(packet.result.original, 200)))
    monkeypatch.setattr(ui.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path))
    window.export_snapshot()
    import cv2
    exported = next(tmp_path.glob("subtitle-compare-*/original.png"))
    image = cv2.imdecode(np.fromfile(exported, np.uint8), cv2.IMREAD_COLOR)
    assert np.all(image == 60)
