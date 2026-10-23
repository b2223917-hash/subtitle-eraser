"""Render our own Qt windows using synthetic data, without touching the desktop.

Run in separate processes with QT_SCREEN_SCALE_FACTORS=1.25 / 1.5 / 2;
for 100% on a scaled Windows desktop set QT_ENABLE_HIGHDPI_SCALING=0.
The JSON records the measured DPR, rather than assuming the override applied.
"""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from subtitle_eraser import __version__, ui
from subtitle_eraser.fixtures import sample_frame
from subtitle_eraser.pipeline import Settings, SubtitleProcessor
from subtitle_eraser.profiles import ProfileStore
from subtitle_eraser.worker import Packet


def check(output):
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(ui.STYLE)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    with TemporaryDirectory(prefix="subtitle-ui-check-") as temporary, ExitStack() as stack:
        for name in ("register_toggle", "register_selection", "register_selection_fallback",
                     "exclude_from_capture"):
            stack.enter_context(patch.object(ui.windows, name, return_value=True))
        for name in ("unregister_toggle", "unregister_selection"):
            stack.enter_context(patch.object(ui.windows, name, return_value=None))
        stack.enter_context(patch.object(ui.windows, "list_windows", return_value=[]))
        window = ui.MainWindow(ProfileStore(Path(temporary)/"profiles.json"))
        window.timer.stop()
        comparison = window.comparison
        for widget in (window, comparison):
            widget.setAttribute(Qt.WA_DontShowOnScreen)
        frame, *_ = sample_frame(8)
        result = SubtitleProcessor(settings=Settings(mode="manual")).process(frame[351:443, 96:864])
        packet = Packet(1, result, perf_counter(), perf_counter(), (960, 540), (96, 351, 768, 92), 0)
        worker = SimpleNamespace(epoch=0, roi=(.1, .65, .8, .17), source=None,
                                 snapshot=lambda: ((frame, packet.captured_at), packet),
                                 isRunning=lambda: True)
        window.worker = worker
        window.window_combo.addItem("演示播放器（合成画面）")
        window.source_label.setText("界面检查 · 程序生成的字幕样例，不采集其他窗口")
        window.show()
        window.poll()

        def settle():
            for _ in range(4):
                app.processEvents()

        settle()
        records = {}
        for name, widget in (("main", window), ("comparison", comparison)):
            screen = widget.screen().availableGeometry().size()
            assert widget.width() <= screen.width() and widget.height() <= screen.height(), name
            records[name] = {"logical_size": [widget.width(), widget.height()],
                             "screen_available": [screen.width(), screen.height()]}
        window.grab().save(str(output/"main.png"))
        window.advanced_toggle.click()
        settle()
        assert window.controls_scroll.horizontalScrollBar().maximum() == 0
        window.grab().save(str(output/"main-advanced.png"))
        window.advanced_toggle.click()
        window.open_comparison()
        settle()
        for preview, caption in zip(comparison.previews, comparison.captions):
            rect = preview.rect().translated(preview.mapTo(comparison.scroll.viewport(), preview.rect().topLeft()))
            assert comparison.scroll.viewport().rect().contains(rect), "Preview clipped"
            assert not preview.geometry().intersects(caption.geometry()), "Caption overlaps preview"
            assert preview.rect().contains(preview.image_rect().toAlignedRect()), "Image cropped"
        comparison.grab().save(str(output/"comparison-all.png"))
        comparison.view_tabs.setCurrentIndex(3)
        settle()
        comparison.grab().save(str(output/"comparison-c.png"))
        comparison.view_tabs.setCurrentIndex(0)
        comparison.orientation.setCurrentIndex(1)
        settle()
        comparison.grab().save(str(output/"comparison-vertical.png"))
        comparison.close()
        assert window.worker is worker, "Closing comparison stopped processing"
        data = {"passed": True, "version": __version__, "dpr": window.devicePixelRatioF(),
                "windows": records, "synthetic_data_only": True}
        (output/"layout.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(data, ensure_ascii=False))
        window.worker = None
        window.close()
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/ui-layout"))
    raise SystemExit(check(parser.parse_args().output))
