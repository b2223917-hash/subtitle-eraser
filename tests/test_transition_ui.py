"""Exercise presentation guards without showing windows or capturing the desktop."""
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from subtitle_eraser import ui
from subtitle_eraser.pipeline import FrameResult, Settings
from subtitle_eraser.worker import Packet


class HiddenOverlay(ui.RepairOverlay):
    """Use the real QImage conversion, but never show its native window."""

    def __init__(self):
        self.presented = False
        self.events = []
        self.compositions = 0
        super().__init__()

    def show(self):
        self.events.append(("show",))
        self.presented = True

    def hide(self):
        self.events.append(("hide",))
        self.presented = False

    def isVisible(self):
        return self.presented

    def set_result(self, repaired, mask, *, cover_region=False):
        super().set_result(repaired, mask, cover_region=cover_region)
        self.compositions += 1


class FakeSource:
    def __init__(self, frame, captured_at):
        self.frame = frame
        self.captured_at = captured_at
        self.closed = False

    def read(self):
        return 1, self.frame, self.captured_at


class FakeWorker:
    def __init__(self, frame, packet):
        self.latest_frame = frame, packet.captured_at
        self.latest_packet = packet
        self.epoch = packet.epoch
        self.running = True
        self.source = FakeSource(frame, packet.captured_at)
        self.roi = (.1, .6, .8, .2)
        self.settings = Settings(mode="manual")

    def isRunning(self):
        return self.running

    def snapshot(self):
        return self.latest_frame, self.latest_packet

    def configure(self, roi, settings):
        self.roi = roi
        self.settings = settings
        self.epoch += 1
        self.latest_packet = None


@pytest.fixture
def presentation(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui, "RepairOverlay", HiddenOverlay)
    monkeypatch.setattr(ui.windows, "list_windows", lambda: [])
    monkeypatch.setattr(ui.windows, "register_toggle", lambda _: True)
    monkeypatch.setattr(ui.windows, "unregister_toggle", lambda _: None)
    monkeypatch.setattr(ui.windows, "exclude_from_capture", lambda _: True)
    state = SimpleNamespace(bounds=(100, 200, 120, 80), foreground=1234, available=True)
    monkeypatch.setattr(ui.windows, "bounds", lambda _: state.bounds)
    monkeypatch.setattr(ui.windows, "foreground_root", lambda: state.foreground)
    monkeypatch.setattr(ui.windows, "available", lambda _: state.available)
    window = ui.MainWindow()
    window.timer.stop()
    monkeypatch.setattr(ui.windows, "position_overlay",
                        lambda handle, rect: window.overlay.events.append(("position", rect)))
    frame = np.full((80, 120, 3), (24, 68, 113), np.uint8)
    original = frame[50:68, 10:100].copy()
    result = FrameResult(original, np.zeros((18, 90), np.uint8), original.copy(), [],
                         "当前无字幕", {"total_ms": 1.0})
    now = perf_counter()
    packet = Packet(42, result, now, now + .001, (120, 80), (10, 50, 90, 18), 3)
    worker = FakeWorker(frame, packet)
    window.worker = worker
    window.window_info = SimpleNamespace(hwnd=1234)
    state.window, state.worker, state.packet = window, worker, packet
    try:
        # These widgets are deliberately never shown. All presentation uses the
        # recording overlay, and all native window positioning is mocked above.
        yield state
    finally:
        window.worker = None
        window.close()
        window.overlay.deleteLater()
        window.deleteLater()
        app.sendPostedEvents()


def rgba(overlay):
    image = overlay.image
    rows = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return rows[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


def test_blank_frame_stays_opaque_between_subtitles(presentation):
    state = presentation
    window = state.window
    window.poll()
    assert window.overlay.isVisible()
    pixels = rgba(window.overlay)
    assert np.all(pixels[:, :, 3] == 255)
    assert np.array_equal(pixels[:, :, :3], state.packet.result.repaired[:, :, ::-1])


def test_glyph_only_mode_hides_an_empty_mask(presentation):
    window = presentation.window
    window.poll()
    window.continuity_check.setChecked(False)
    assert not window.overlay.isVisible()
    assert not np.any(rgba(window.overlay)[:, :, 3])


def test_presentation_toggle_recomposes_same_static_packet_immediately(presentation):
    window, packet = presentation.window, presentation.packet
    packet.static_source = True
    packet.result.mask[4:12, 20:45] = 255
    window.poll()
    first_compositions = window.overlay.compositions
    window.continuity_check.setChecked(False)
    assert window.overlay.compositions == first_compositions + 1
    assert window.last_sequence == packet.sequence
    assert np.array_equal(rgba(window.overlay)[:, :, 3], packet.result.mask)
    assert window.overlay.isVisible()
    window.continuity_check.setChecked(True)
    assert window.overlay.compositions == first_compositions + 2
    assert np.all(rgba(window.overlay)[:, :, 3] == 255)
    assert window.overlay.isVisible()


@pytest.mark.parametrize("reason", [
    "paused", "unchecked", "resized", "unfocused", "stale", "epoch",
    "closed", "failed", "unavailable", "stopped", "no_window", "no_packet",
])
def test_invalid_presentations_hide_previous_overlay(presentation, reason):
    state = presentation
    window, worker, packet = state.window, state.worker, state.packet
    window.poll()
    assert window.overlay.isVisible()
    if reason == "paused":
        window.paused = True
    elif reason == "unchecked":
        window.live_check.setChecked(False)
    elif reason == "resized":
        state.bounds = (100, 200, 130, 80)
    elif reason == "unfocused":
        state.foreground = 5678
    elif reason == "stale":
        worker.source.captured_at = packet.captured_at + .351
    elif reason == "epoch":
        worker.epoch += 1
    elif reason == "closed":
        worker.source.closed = True
    elif reason == "failed":
        packet.result.safe_to_present = False
        packet.result.status = "文字轮廓不可靠，已保留原图"
        packet.sequence += 1
    elif reason == "unavailable":
        state.available = False
    elif reason == "stopped":
        worker.running = False
        window.status.setText("处理已停止：采集失败")
        packet.sequence += 1
    elif reason == "no_window":
        window.window_info = None
    elif reason == "no_packet":
        worker.latest_packet = None
    window.poll()
    assert not window.overlay.isVisible()
    if reason == "failed":
        assert window.status.text() == packet.result.status
    elif reason == "stopped":
        assert window.status.text() == "处理已停止：采集失败"
    elif reason == "resized":
        assert "自动定位" in window.status.text()
    elif reason == "stale":
        assert "跟不上" in window.status.text()


def test_geometry_is_synced_before_show_and_native_position_after_show(presentation):
    window = presentation.window
    window.overlay.events.clear()
    window.poll()
    assert window.overlay.events == [("show",), ("position", (110, 250, 90, 18))]
    window.overlay.events.clear()
    window.poll()
    assert not window.overlay.events
    presentation.bounds = (125, 240, 120, 80)
    window.poll()
    assert window.overlay.events == [("position", (135, 290, 90, 18))]
    window.overlay.hide()
    window.overlay.events.clear()
    window.poll()
    assert window.overlay.events == [("show",), ("position", (135, 290, 90, 18))]


def test_auto_mode_disables_continuous_crop_and_invalidates_manual_selection(presentation):
    window = presentation.window
    window.poll()
    window.mode_combo.setCurrentIndex(1)
    assert not window.cover_region()
    assert not window.continuity_check.isEnabled()
    assert not window.overlay.isVisible()
    assert presentation.worker.roi is None
    assert presentation.worker.latest_packet is None
