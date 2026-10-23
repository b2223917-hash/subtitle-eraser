"""Interactive native check. Send the printed shortcut to the generated player.

Creates only its own windows. No input is injected and no other app is captured.
The operator/Computer Use sends the real key chord, then cancels the selector.
"""
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from time import perf_counter

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication
from subtitle_eraser import windows
from subtitle_eraser.diagnostics import transition_fixture
from subtitle_eraser.profiles import ProfileStore
from subtitle_eraser.ui import MainWindow, SelectionDialog, qimage
from subtitle_eraser.worker import ProcessingWorker


def run(output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    from PySide6.QtWidgets import QWidget
    before, after, roi, _ = transition_fixture()

    class Player(QWidget):
        def __init__(self, image, title):
            super().__init__(None, Qt.FramelessWindowHint)
            self.image = qimage(image)
            self.setWindowTitle(title)
            self.resize(image.shape[1], image.shape[0])

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), Qt.black)
            from PySide6.QtCore import QRect
            size = self.image.size().scaled(self.size(), Qt.KeepAspectRatio)
            painter.drawImage(QRect((self.width()-size.width())//2, (self.height()-size.height())//2,
                                   size.width(), size.height()), self.image)

    temporary = TemporaryDirectory(prefix='subtitle-hotkey-')
    window = MainWindow(profile_store=ProfileStore(Path(temporary.name)/'profiles.json'))
    original = Player(before, 'SubtitleEraser Hotkey Test Original')
    fullscreen = Player(after, 'SubtitleEraser Hotkey Test Player')
    window.setWindowTitle('SubtitleEraser Hotkey Test Controller')
    window.show()
    original.show()
    info = windows.window_info(int(original.winId()))
    window.window_info = info
    window.player_identity = windows.player_identity(info.hwnd)
    worker = ProcessingWorker(window=info)
    window.worker = worker
    worker.packet_ready.connect(window.on_packet_ready, Qt.QueuedConnection)
    worker.start()
    started = perf_counter()
    state = {'passed': False, 'shortcut': window.selection_shortcut,
             'generated_windows_only': True, 'hotkey_received': 0, 'dialog_visible': False,
             'original_hwnd_destroyed': False, 'fullscreen_followed': False, 'overlay_restored': False}
    stage = 'first-frame'

    def record_hotkey(action):
        if action == 2:
            state['hotkey_received'] += 1

    window.hotkey_events.activated.connect(record_hotkey)

    def save():
        output.write_text(json.dumps(state, indent=2), encoding='utf-8')

    def tick():
        nonlocal stage
        frame, packet = worker.snapshot()
        if perf_counter()-started > 120:
            state['timeout_stage'] = stage
            save()
            dialog = QApplication.activeModalWidget()
            if dialog is not None:
                dialog.reject()
            app.exit(1)
            return
        if stage == 'first-frame' and frame is not None:
            worker.configure(roi, window.settings())
            stage = 'first-packet'
        elif stage == 'first-packet' and packet is not None:
            fullscreen.showFullScreen()
            original.close()
            original.destroy()
            state['original_hwnd_destroyed'] = not bool(windows.user32.IsWindow(info.hwnd))
            stage = 'fullscreen-follow'
        elif stage == 'fullscreen-follow' and packet is not None:
            if worker.window.hwnd == int(fullscreen.winId()) and packet.frame_size == windows.bounds(int(fullscreen.winId()))[2:]:
                state['fullscreen_followed'] = True
                stage = 'await-real-key'
                save()
                print(json.dumps(state), flush=True)
        elif stage == 'await-real-key':
            dialog = QApplication.activeModalWidget()
            if isinstance(dialog, SelectionDialog):
                state['dialog_visible'] = dialog.isVisible()
                state['dialog_hwnd'] = int(dialog.winId())
                stage = 'await-cancel'
                save()
                QTimer.singleShot(6000, dialog.reject)
        elif stage == 'await-cancel' and QApplication.activeModalWidget() is None:
            fullscreen.activateWindow()
            stage = 'restore-overlay'
        elif stage == 'restore-overlay' and window.overlay.isVisible():
            state['overlay_restored'] = True
            state['passed'] = all((state['hotkey_received'] > 0, state['dialog_visible'],
                state['original_hwnd_destroyed'], state['fullscreen_followed'], state['overlay_restored']))
            save()
            app.exit(0 if state['passed'] else 1)

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(40)
    result = app.exec()
    timer.stop()
    window.close()
    original.close()
    fullscreen.close()
    temporary.cleanup()
    return result

