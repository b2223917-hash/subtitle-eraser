"""A separate, deterministic player window for WGC integration QA."""
from time import perf_counter
from PySide6.QtWidgets import QApplication, QLabel
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap
from .ui import qimage
from .fixtures import sample_frame

app=QApplication([])
window=QLabel()
window.setWindowTitle("留英文 · 捕获测试播放器")
window.resize(960,540)
window.setAlignment(Qt.AlignCenter)
window.setStyleSheet("background:#111c2e")
started=perf_counter()


def update():
    # Loop the simple scene so subtitle calibration is repeatable.
    frame,*_=sample_frame(int((perf_counter()-started)*24)%38)
    window.setPixmap(QPixmap.fromImage(qimage(frame)).scaled(window.size(),Qt.KeepAspectRatio))


timer=QTimer()
timer.timeout.connect(update)
timer.start(42)
update()
window.show()
app.exec()
