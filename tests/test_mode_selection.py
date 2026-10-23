"""Switching modes must not turn a bilingual crop into an erase-English crop."""
from PySide6.QtWidgets import QApplication, QPushButton
import numpy as np

from subtitle_eraser.ui import MainWindow, SelectionDialog
from subtitle_eraser.worker import ProcessingWorker


def test_switching_mode_requires_new_language_boundary():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.timer.stop()
    try:
        assert window.settings().mode == "manual"
        worker = ProcessingWorker(demo=True)
        window.worker = worker
        worker.configure((.1, .6, .8, .35), window.settings())
        window.mode_combo.setCurrentIndex(1)
        assert worker.roi is None
        assert worker.settings.mode == "auto"
        worker.configure((.1, .6, .8, .35), window.settings())
        worker.latest_packet = object()
        window.mode_combo.setCurrentIndex(0)
        assert worker.roi is None
        assert worker.latest_packet is None
        assert worker.settings.mode == "manual"
    finally:
        window.worker = None
        window.close()


def test_manual_selection_does_not_offer_bilingual_bottom_preset():
    app = QApplication.instance() or QApplication([])
    dialog = SelectionDialog(np.zeros((100, 300, 3), np.uint8), None, manual=True)
    try:
        assert not any(b.text() == "选中底部区域" for b in dialog.findChildren(QPushButton))
        assert not dialog.ok.isEnabled()
    finally:
        dialog.close()
