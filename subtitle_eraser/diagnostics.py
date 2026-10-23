"""Checks of our own native windows; never reads another application's pixels."""
from time import perf_counter, sleep
import json
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication, QWidget
from . import windows


def native_geometry_check():
    from .ui import RepairOverlay
    app = QApplication.instance() or QApplication([])
    overlay = RepairOverlay()
    checks = []
    try:
        for index, rect in enumerate([(-4000, -4000, 768, 60), (-3900, -3900, 900, 42),
                                       (-4000, -3900, 750, 48), (-3900, -4000, 769, 61)]):
            if index == 2:
                overlay.hide()
            image = np.full((rect[3], rect[2], 3), 48, np.uint8)
            overlay.set_result(image, np.zeros(image.shape[:2], np.uint8), cover_region=True)
            overlay.present_at(rect)
            app.processEvents()
            actual = windows.bounds(int(overlay.winId()))
            assert max(abs(a-b) for a, b in zip(actual, rect)) <= 1, (rect, actual)
            assert overlay.isVisible()
            checks.append({"expected": rect, "actual": actual, "dpr": overlay.devicePixelRatioF()})
    finally:
        overlay.close()
        overlay.deleteLater()
        app.processEvents()
    return checks


def transition_fixture():
    rng = np.random.default_rng(47)
    video = np.full((300, 500, 3), 40, np.uint8)
    for _ in range(160):
        x, y = rng.integers([0, 0], [500, 220])
        cv2.circle(video, (int(x), int(y)), int(rng.integers(2, 9)),
                   tuple(int(v) for v in rng.integers(40, 180, 3)), -1)
    cv2.putText(video, 'SUBTITLE TEST 123', (35, 255), cv2.FONT_HERSHEY_DUPLEX, .8, (250, 250, 250), 2)
    cv2.putText(video, 'English stays below', (70, 284), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1)
    before = np.full((380, 740, 3), (15, 35, 60), np.uint8)
    before[30:330, 10:510] = video
    cv2.putText(before, 'SIDEBAR', (540, 90), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    after = cv2.resize(video, (1000, 600), interpolation=cv2.INTER_LINEAR)
    return before, after, (30/740, 261/380, 460/740, 30/380), (40/1000, 462/600, 920/1000, 60/600)


def native_transition_check(output):
    """Briefly show our generated test player and capture only its HWND.

    Includes a real fullscreen HWND, normal resize, return and opaque overlay
    placement at OS DPI. No foreground interaction or user video is required.
    """
    from .ui import qimage, RepairOverlay
    from .worker import ProcessingWorker
    from .pipeline import Settings
    app = QApplication.instance() or QApplication([])
    before, fullscreen, roi, expected = transition_fixture()

    class Player(QWidget):
        def __init__(self, frame):
            super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowDoesNotAcceptFocus)
            self.setAttribute(Qt.WA_ShowWithoutActivating)
            self.setWindowTitle("SubtitleEraser — generated window test")
            self.frame = frame
            self.setGeometry(20, 20, frame.shape[1], frame.shape[0])

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.black)
            image = qimage(self.frame)
            size = image.size().scaled(self.size(), Qt.KeepAspectRatio)
            from PySide6.QtCore import QRect
            painter.drawImage(QRect((self.width()-size.width())//2, (self.height()-size.height())//2,
                                    size.width(), size.height()), image)

    player, other, overlay = Player(before), Player(fullscreen), RepairOverlay()
    player.show()
    app.processEvents()
    original = windows.WindowInfo(int(player.winId()), player.windowTitle())
    worker = ProcessingWorker(window=original)
    worker.start()
    checks = []
    try:
        for stage in ("windowed", "resized", "restored", "fullscreen_hwnd", "return_hwnd"):
            expected_roi = roi
            target = player
            overlay.hide()
            if stage == "resized":
                player.frame = fullscreen
                player.resize(fullscreen.shape[1], fullscreen.shape[0])
                expected_roi = expected
                player.update()
            elif stage == "restored":
                player.frame = before
                player.resize(before.shape[1], before.shape[0])
                player.update()
            elif stage == "fullscreen_hwnd":
                other.showFullScreen()
                app.processEvents()
                assert windows.is_fullscreen(int(other.winId()))
                target = other
                worker.request_window(windows.WindowInfo(int(other.winId()), other.windowTitle()))
                # Map the fixture through the test player's letterboxed paint.
                tw, th = windows.bounds(int(other.winId()))[2:]
                scale = min(tw/1000, th/600)
                expected_roi = ((tw-1000*scale)/2/tw+expected[0]*1000*scale/tw,
                                (th-600*scale)/2/th+expected[1]*600*scale/th,
                                expected[2]*1000*scale/tw, expected[3]*600*scale/th)
            elif stage == "return_hwnd":
                other.hide()
                worker.request_window(original)
            app.processEvents()
            started, configured, packet = perf_counter(), False, None
            while perf_counter()-started < 10:
                app.processEvents()
                sleep(.01)
                frame, packet = worker.snapshot()
                if stage == "windowed" and frame is not None and not configured:
                    worker.configure(roi, Settings(mode="manual"))
                    configured = True
                if packet is not None and packet.frame_size == windows.bounds(int(target.winId()))[2:]:
                    break
            elapsed = (perf_counter()-started)*1000
            assert packet is not None, f"No repaired packet after {stage}"
            fw, fh = packet.frame_size
            expected_pixels = np.array(expected_roi)*[fw, fh, fw, fh]
            assert np.max(np.abs(np.array(packet.roi)-expected_pixels)) <= 3, (stage, packet.roi, expected_pixels)
            x, y = windows.bounds(int(target.winId()))[:2]
            rx, ry, rw, rh = packet.roi
            overlay.set_result(packet.result.repaired, packet.result.mask, cover_region=True)
            rect = (x+rx, y+ry, rw, rh)
            overlay.present_at(rect)
            app.processEvents()
            actual = windows.bounds(int(overlay.winId()))
            assert max(abs(a-b) for a, b in zip(actual, rect)) <= 1, (stage, actual, rect)
            checks.append({"stage": stage, "recovery_ms": round(elapsed, 1), "frame_size": packet.frame_size,
                           "roi": packet.roi, "overlay": actual, "dpr": overlay.devicePixelRatioF()})
    finally:
        overlay.close()
        worker.stop()
        assert worker.wait(5000), "Test worker did not stop"
        player.close()
        other.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": True, "generated_windows_only": True, "checks": checks}, indent=2), encoding="utf-8")
    return 0


def native_selection_check(output):
    """Exercise the real modal selector, resize-on-focus and full Chinese repair."""
    from tempfile import TemporaryDirectory
    from PySide6.QtCore import QRect, QTimer
    from .ui import MainWindow, SelectionDialog, qimage
    from .fixtures import sample_frame
    from .profiles import ProfileStore
    from .worker import ProcessingWorker
    app = QApplication.instance() or QApplication([])
    source, _, truth, _ = sample_frame(0)

    class Player(QWidget):
        def __init__(self):
            super().__init__(None, Qt.FramelessWindowHint | Qt.Tool)
            self.setWindowTitle("SubtitleEraser — Chinese fullscreen selection test")

        def paintEvent(self, event):
            p = QPainter(self)
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.fillRect(self.rect(), Qt.black)
            size = qimage(source).size().scaled(self.size(), Qt.KeepAspectRatio)
            p.drawImage(QRect((self.width()-size.width())//2, (self.height()-size.height())//2,
                             size.width(), size.height()), qimage(source))

    def wait_until(predicate, seconds=8):
        start = perf_counter()
        while perf_counter()-start < seconds:
            app.processEvents()
            if predicate():
                return
            sleep(.01)
        raise AssertionError("Native selection check timed out")

    player = Player()
    temporary = TemporaryDirectory(prefix="subtitle-native-selection-")
    window = MainWindow(profile_store=ProfileStore(__import__('pathlib').Path(temporary.name)/'profiles.json'))
    player.showFullScreen()
    app.processEvents()
    target = windows.WindowInfo(int(player.winId()), player.windowTitle())
    worker = ProcessingWorker(window=target)
    window.worker, window.window_info = worker, target
    worker.start()
    try:
        full_size = windows.bounds(target.hwnd)[2:]
        wait_until(lambda: worker.snapshot()[0] is not None and worker.snapshot()[0][0].shape[1::-1] == full_size)
        fw, fh = full_size
        factor = min(fw/source.shape[1], fh/source.shape[0])
        vw, vh = round(source.shape[1]*factor), round(source.shape[0]*factor)
        dx, dy = (fw-vw)//2, (fh-vh)//2
        mapped_truth = np.zeros((fh, fw), np.uint8)
        mapped_truth[dy:dy+vh, dx:dx+vw] = cv2.resize(truth, (vw, vh))
        x, y, w, h = cv2.boundingRect(mapped_truth)
        # Deliberately choose only the middle, matching the user's missing ends.
        roi = ((x+w*.22)/fw, (y-5)/fh, w*.56/fw, (h+10)/fh)
        callbacks = []

        def change_layout():
            dialog = QApplication.activeModalWidget()
            if not isinstance(dialog, SelectionDialog):
                callbacks.append("selector missing")
                return
            player.showNormal()
            player.resize(740, 440)
            player.move(20, 20)
            dialog.canvas.set_normalized_region(roi)

        def accept_when_resized():
            dialog = QApplication.activeModalWidget()
            if not isinstance(dialog, SelectionDialog):
                return
            frame = worker.snapshot()[0]
            if frame is None or frame[0].shape[1::-1] == full_size:
                QTimer.singleShot(50, accept_when_resized)
                return
            callbacks.append("accepted after resize")
            dialog.accept()

        QTimer.singleShot(120, change_layout)
        QTimer.singleShot(500, accept_when_resized)
        # A bounded escape also ensures a failed test never strands a modal dialog.
        QTimer.singleShot(5000, lambda: QApplication.activeModalWidget().reject()
                          if QApplication.activeModalWidget() is not None else None)
        window.select_region()
        assert "accepted after resize" in callbacks, callbacks
        assert "已保存" in window.preference_note.text(), window.preference_note.text()
        started = perf_counter()
        player.showFullScreen()
        player.activateWindow()
        wait_until(lambda: worker.snapshot()[1] is not None and worker.snapshot()[1].frame_size == full_size)
        packet = worker.snapshot()[1]
        assert packet.result.safe_to_present
        px, py, pw, ph = packet.roi
        whole_mask = np.zeros((fh, fw), np.uint8)
        whole_mask[py:py+ph, px:px+pw] = packet.result.mask
        coverage = float(np.mean(whole_mask[mapped_truth > 8] > 0))
        assert coverage > .99, coverage
        window.poll()
        try:
            wait_until(lambda: window.overlay.isVisible(), 2)
        except AssertionError:
            pass  # Report all native presentation guards below.
        assert window.overlay.isVisible(), {"status": window.status.text(), "target": target.hwnd,
            "foreground": windows.foreground_root(), "foreground_pid": windows.process_id(windows.foreground_root()),
            "target_pid": windows.process_id(target.hwnd), "available": windows.available(target.hwnd),
            "native_size": windows.bounds(target.hwnd), "packet": packet.frame_size,
            "epoch": (packet.epoch, worker.epoch), "paused": window.paused,
            "worker_target": worker.window.hwnd, "source_closed": worker.source.closed,
            "latest_capture": worker.source.read()[2]-packet.captured_at}
        data = {"passed": True, "generated_windows_only": True, "selection_survived_focus_resize": True,
                "frame_size": full_size, "roi": packet.roi, "chinese_coverage": coverage,
                "recovery_ms": round((perf_counter()-started)*1000, 1),
                "repair_ms": packet.result.timing['total_ms'], "overlay_visible": True}
    finally:
        window.close()
        player.close()
        temporary.cleanup()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return 0
