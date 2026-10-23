from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
from time import perf_counter
import json
import os
import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt, QRect, QRectF, QPoint, QSize, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QColor, QFont, QFontDatabase, QImage, QPainter, QPen
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog,
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPushButton, QRubberBand, QScrollArea, QSlider, QSizePolicy,
    QSpinBox, QTabBar, QTextBrowser, QVBoxLayout, QWidget)

from . import windows
from . import __version__
from .pipeline import Settings, SubtitleProcessor
from .style import calibrate_style, SubtitleStyle
from .profiles import ProfileStore
from .worker import ProcessingWorker
from .presentation import overlay_rgba
from .hotkeys import HotkeyEvents, HotkeyFilter

ROOT = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent


def qimage(image: np.ndarray) -> QImage:
    image = np.ascontiguousarray(image)
    if image.ndim == 2:
        return QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_Grayscale8).copy()
    return QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format_BGR888).copy()


def fit_window(window, preferred, minimum):
    """Use logical screen coordinates, including Windows display scaling."""
    screen = window.screen() or QApplication.primaryScreen()
    available = screen.availableGeometry().size() if screen else QSize(*preferred)
    width, height = max(320, available.width()-32), max(300, available.height()-56)
    window.setMinimumSize(min(minimum[0], width), min(minimum[1], height))
    window.resize(min(preferred[0], width), min(preferred[1], height))


class Preview(QWidget):
    double_clicked = Signal()

    def __init__(self, text="框选字幕后，画面会显示在这里"):
        super().__init__()
        self.image = None
        self.placeholder = text
        self.setMinimumSize(80, 96)
        # Wrapped captions contribute height-for-width to the enclosing cards.
        # Ignore the preferred image height there, so a small viewport uses the
        # real minimum rather than needlessly scrolling a 220px image hint.
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.setToolTip("画面按比例完整显示；双击可切换单图放大。")

    def sizeHint(self):
        return QSize(300, 220)

    def set_image(self, image):
        self.image = qimage(image)
        self.update()

    def clear(self, text=None):
        self.image = None
        if text is not None:
            self.placeholder = text
        self.update()

    def image_rect(self):
        bounds = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        if self.image is None:
            return bounds
        size = self.image.size().scaled(bounds.size().toSize(), Qt.KeepAspectRatio)
        return QRectF(bounds.center().x()-size.width()/2, bounds.center().y()-size.height()/2,
                      size.width(), size.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#111c2e"))
        painter.drawRoundedRect(self.rect(), 10, 10)
        painter.setClipRect(self.rect().adjusted(8, 8, -8, -8))
        if self.image is None:
            painter.setPen(QColor("#9baac0"))
            painter.drawText(self.rect().adjusted(14, 10, -14, -10),
                             Qt.AlignCenter | Qt.TextWordWrap, self.placeholder)
        else:
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawImage(self.image_rect(), self.image)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)


class ComparisonWindow(QMainWindow):
    refresh_requested = Signal()

    def __init__(self, parent):
        # A native owned window is minimized along with its owner on Windows.
        # Keep this top level independent so controls can be minimized alone.
        super().__init__()
        self.cached_frame = self.cached_packet = None
        self.setWindowTitle(f"留英文 · 效果对比 A / B / C · {__version__}")
        fit_window(self, (1120, 700), (640, 440))
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        title = QLabel("效果对比")
        title.setStyleSheet("font-size:22px;font-weight:700;color:#172843;")
        layout.addWidget(title)
        toolbar = QHBoxLayout()
        self.view_tabs = QTabBar()
        self.view_tabs.setExpanding(False)
        self.view_tabs.setDrawBase(False)
        for label in ("全部", "A 原始", "B 轮廓", "C 修补"):
            self.view_tabs.addTab(label)
        toolbar.addWidget(self.view_tabs)
        toolbar.addStretch()
        self.orientation = QComboBox()
        self.orientation.addItems(("横向并排", "纵向排列"))
        self.orientation.setToolTip("三张画面的排列方式；空间不足时可滚动查看。")
        toolbar.addWidget(self.orientation)
        self.freeze_button = QPushButton("冻结对比")
        self.freeze_button.setCheckable(True)
        self.freeze_button.setToolTip("只冻结对比画面，原播放器的实时修补继续运行。")
        toolbar.addWidget(self.freeze_button)
        layout.addLayout(toolbar)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        self.panels_layout = QHBoxLayout(content)
        self.panels_layout.setContentsMargins(0, 0, 0, 0)
        self.panels_layout.setSpacing(12)
        self.previews, self.panels, self.captions = [], [], []
        for index, (name, note) in enumerate((("A  原始字幕", "当前选区的原画面"),
                                             ("B  文字轮廓", "绿色：修补区域；蓝框：保护区域"),
                                             ("C  局部修补", "当前选区的处理结果"))):
            panel = QFrame()
            panel.setObjectName("card")
            panel.setMinimumWidth(170)
            column = QVBoxLayout(panel)
            column.setContentsMargins(12, 12, 12, 12)
            heading = QLabel(name)
            heading.setStyleSheet("font-size:15px;font-weight:600;")
            column.addWidget(heading)
            preview = Preview("等待字幕画面\n连接来源并框选后显示")
            preview.double_clicked.connect(lambda i=index: self.toggle_focus(i))
            column.addWidget(preview, 1)
            caption = QLabel(note)
            caption.setObjectName("muted")
            caption.setWordWrap(True)
            column.addWidget(caption)
            self.panels_layout.addWidget(panel, 1)
            self.previews.append(preview)
            self.panels.append(panel)
            self.captions.append(caption)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll, 1)
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.metric_labels = []
        for name, tip in (("处理耗时", "当前选区轮廓提取与修补的耗时。"),
                          ("画面更新延迟", "从收到捕获画面到界面更新的时间，不含捕获前的等待或屏幕扫描。"),
                          ("更新速度", "处理结果的更新速度；对比预览最多每秒刷新 10 次。"),
                          ("修补占比", "当前选区内被修补的像素比例。")):
            card = QFrame()
            card.setObjectName("card")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 9, 12, 9)
            label = QLabel(name)
            label.setObjectName("muted")
            value = QLabel("—")
            value.setStyleSheet("font-size:21px;font-weight:600;color:#243d6a;")
            value.setToolTip(tip)
            card_layout.addWidget(label)
            card_layout.addWidget(value)
            metrics.addWidget(card, 1)
            self.metric_labels.append(value)
        layout.addLayout(metrics)
        footer = QHBoxLayout()
        self.hint = QLabel("双击画面放大；关闭对比窗口后，播放器修补继续运行。")
        self.hint.setWordWrap(True)
        self.hint.setObjectName("muted")
        self.hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        footer.addWidget(self.hint, 1)
        self.export_button = QPushButton("保存对比快照")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(parent.export_snapshot)
        footer.addWidget(self.export_button)
        layout.addLayout(footer)
        self.setCentralWidget(central)
        self.view_tabs.currentChanged.connect(self.update_view)
        self.orientation.currentIndexChanged.connect(self.update_view)
        self.freeze_button.toggled.connect(self.change_freeze)

    @property
    def frozen(self):
        return self.freeze_button.isChecked()

    def change_freeze(self, frozen):
        self.freeze_button.setText("继续对比" if frozen else "冻结对比")
        self.hint.setText("对比画面已冻结；实时修补与性能统计继续更新。" if frozen else
                          "双击画面放大；关闭对比窗口后，播放器修补继续运行。")
        if not frozen:
            self.refresh_requested.emit()

    def toggle_focus(self, index):
        self.view_tabs.setCurrentIndex(0 if self.view_tabs.currentIndex() == index+1 else index+1)

    def update_view(self):
        selected = self.view_tabs.currentIndex()-1
        self.orientation.setEnabled(selected == -1)
        vertical = selected == -1 and self.orientation.currentIndex() == 1
        direction = QVBoxLayout.TopToBottom if vertical else QHBoxLayout.LeftToRight
        self.panels_layout.setDirection(direction)
        for index, panel in enumerate(self.panels):
            self.previews[index].setMinimumHeight(48 if vertical else 96)
            panel.setVisible(selected in (-1, index))
        self.refresh_requested.emit()

    def clear(self):
        self.cached_frame = self.cached_packet = None
        for preview in self.previews:
            preview.clear("等待字幕画面\n连接来源并框选后显示")
        for label in self.metric_labels:
            label.setText("—")
        self.freeze_button.setChecked(False)
        self.export_button.setEnabled(False)

    def showEvent(self, event):
        super().showEvent(event)
        self.refresh_requested.emit()


class SelectionCanvas(QWidget):
    changed = Signal()

    def __init__(self, image):
        super().__init__()
        self.image = qimage(image)
        self.setMinimumSize(700, 340)
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)
        self._normalized = None
        self.start = None

    @property
    def selected(self):
        if self._normalized is None:
            return QRectF()
        x, y, w, h = self._normalized
        b = self.image_rect()
        return QRectF(b.x()+x*b.width(), b.y()+y*b.height(), w*b.width(), h*b.height())

    @selected.setter
    def selected(self, rect):
        b = self.image_rect()
        self._normalized = ((rect.x()-b.x())/b.width(), (rect.y()-b.y())/b.height(),
                            rect.width()/b.width(), rect.height()/b.height())

    def set_normalized_region(self, roi):
        self._normalized = tuple(roi)
        self.update()
        self.changed.emit()

    def image_rect(self):
        scale = min(self.width()/self.image.width(), self.height()/self.image.height())
        w, h = self.image.width()*scale, self.image.height()*scale
        return QRectF((self.width()-w)/2, (self.height()-h)/2, w, h)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111c2e"))
        bounds = self.image_rect()
        p.drawImage(bounds, self.image)
        p.fillRect(bounds, QColor(9, 18, 32, 90))
        if not self.selected.isEmpty():
            p.setClipRect(self.selected)
            p.drawImage(bounds, self.image)
            p.setClipping(False)
            p.setPen(QPen(QColor("#70efc0"), 2))
            p.drawRect(self.selected)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image_rect().contains(event.position()):
            self.start = event.position()
            self.selected = QRectF(self.start, self.start)

    def mouseMoveEvent(self, event):
        if self.start is not None:
            self.selected = QRectF(self.start, event.position()).normalized().intersected(self.image_rect())
            self.update()

    def mouseReleaseEvent(self, event):
        if self.start is not None:
            self.mouseMoveEvent(event)
            self.start = None
            self.changed.emit()

    def default_region(self):
        bounds = self.image_rect()
        self.selected = QRectF(bounds.x()+bounds.width()*.10, bounds.y()+bounds.height()*.63,
                               bounds.width()*.80, bounds.height()*.32)
        self.update()
        self.changed.emit()

    def normalized_region(self):
        return self._normalized

    def valid_region(self):
        # A 4K source is heavily reduced in the dialog. Validate source pixels,
        # not the thumbnail's logical-pixel height.
        if self._normalized is None:
            return False
        return self._normalized[2]*self.image.width() >= 20 and self._normalized[3]*self.image.height() >= 12


class SelectionDialog(QDialog):
    def __init__(self, image, parent, manual=False, initial_roi=None):
        super().__init__(parent)
        self.setWindowTitle("只框中文字幕" if manual else "框选双语字幕区域")
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.resize(1000, 680)
        layout = QVBoxLayout(self)
        instruction = ("只框中文行，上下紧贴文字和描边，不要框入英文；左右可留出后续长句的空间。换行或位置变化后重新框选。"
                       if manual else "拖动框选中英文字幕，尽量少包含其他画面。视频改变排版后请重新框选。")
        hint = QLabel(instruction)
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.canvas = SelectionCanvas(image)
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        preset = QPushButton("选中底部区域")
        preset.clicked.connect(self.canvas.default_region)
        # A broad bottom preset would silently include English in manual mode.
        if not manual:
            row.addWidget(preset)
        row.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        row.addWidget(cancel)
        self.ok = QPushButton("使用此区域")
        self.ok.setObjectName("primary")
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self.accept)
        self.canvas.changed.connect(lambda: self.ok.setEnabled(self.canvas.valid_region()))
        row.addWidget(self.ok)
        layout.addLayout(row)
        if initial_roi is not None:
            hint.setText("已标出保存的范围，请检查是否仍只包含中文字幕；可拖动重选。" if manual else
                         "已标出保存的范围，请检查当前字幕位置；可拖动重选。")
            self.canvas.set_normalized_region(initial_roi)


class SampleCanvas(SelectionCanvas):
    clicked = Signal(int, int)

    def __init__(self, image):
        super().__init__(image)
        self.setMinimumSize(700, 170)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#111c2e"))
        painter.drawImage(self.image_rect(), self.image)

    def mousePressEvent(self, event):
        b = self.image_rect()
        if event.button() == Qt.LeftButton and b.contains(event.position()):
            x = min(self.image.width()-1, int((event.position().x()-b.x())/b.width()*self.image.width()))
            y = min(self.image.height()-1, int((event.position().y()-b.y())/b.height()*self.image.height()))
            self.clicked.emit(x, y)

    def mouseMoveEvent(self, event):
        pass

    def mouseReleaseEvent(self, event):
        pass


class StyleCalibrationDialog(QDialog):
    def __init__(self, image, parent=None, edge_padding=3):
        super().__init__(parent)
        self.setWindowTitle("校准字幕样式")
        self.resize(940, 570)
        self.source = image.copy()
        self.style = None
        self.edge_padding = edge_padding
        layout = QVBoxLayout(self)
        hint = QLabel("点击上方截图中一个中文字的明亮笔画内部，避开描边和背景。查看下方绿色轮廓与修补结果，再应用。")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.canvas = SampleCanvas(image)
        self.canvas.clicked.connect(self.pick)
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.mask_preview = Preview("文字轮廓：等待点击采样")
        self.repair_preview = Preview("修补结果：等待点击采样")
        for preview in (self.mask_preview, self.repair_preview):
            preview.setMinimumHeight(140)
            row.addWidget(preview, 1)
        layout.addLayout(row, 1)
        self.note = QLabel("样式仅用于当前中文选区；背景与文字同色时仍可能误选。字幕换色后请重新校准。")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.ok = QPushButton("应用此样式")
        self.ok.setObjectName("primary")
        self.ok.setEnabled(False)
        self.ok.clicked.connect(self.accept)
        buttons.addWidget(self.ok)
        layout.addLayout(buttons)

    def pick(self, x, y):
        self.style = None
        self.ok.setEnabled(False)
        try:
            style = calibrate_style(self.source, (x, y))
            result = SubtitleProcessor(settings=Settings(mode="manual", style=style,
                                       edge_padding=self.edge_padding)).process(self.source)
            if not result.safe_to_present or not np.any(result.mask):
                raise ValueError("当前样式未能提取可靠轮廓，请点击另一个明亮笔画。")
        except ValueError as exc:
            self.mask_preview.set_image(self.source)
            self.repair_preview.set_image(self.source)
            self.note.setText(str(exc))
            return
        self.style = style
        self.mask_preview.set_image(result.debug_image())
        self.repair_preview.set_image(result.repaired)
        self.note.setText("绿色应尽量贴合中文字，不包含英文或背景细节；如不合适，可重新点击。边缘范围沿用主界面的设置。")
        self.ok.setEnabled(True)


class RepairOverlay(QWidget):
    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint |
                         Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.image = None
        self.capture_excluded = windows.exclude_from_capture(int(self.winId()))

    def present_at(self, rect):
        """Synchronize Qt geometry *before* show, then enforce native placement.

        A hidden QWidget ignores native resize notifications. Calling only
        SetWindowPos before show restores its default 640x480 logical size.
        Qt screen origins on Windows remain in desktop coordinates, while sizes
        and offsets within each screen use device-independent pixels.
        """
        x, y, width, height = rect
        screens = QApplication.screens()
        screen = min(screens, key=lambda s: self._screen_distance(s, x+width/2, y+height/2))
        origin = screen.geometry().topLeft()
        ratio = screen.devicePixelRatio()
        if not self.isVisible():
            self.windowHandle().setScreen(screen)
        self.setGeometry(round(origin.x()+(x-origin.x())/ratio),
                         round(origin.y()+(y-origin.y())/ratio),
                         max(1, round(width/ratio)), max(1, round(height/ratio)))
        if not self.isVisible():
            self.show()
        windows.position_overlay(int(self.winId()), rect)

    @staticmethod
    def _screen_distance(screen, x, y):
        g, ratio = screen.geometry(), screen.devicePixelRatio()
        dx = max(g.x()-x, 0, x-(g.x()+g.width()*ratio))
        dy = max(g.y()-y, 0, y-(g.y()+g.height()*ratio))
        return dx*dx + dy*dy

    def set_result(self, repaired, mask, *, cover_region=False):
        rgba = overlay_rgba(repaired, mask, cover_region=cover_region)
        self.image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format_RGBA8888).copy()
        self.update()

    def paintEvent(self, event):
        if self.image is not None:
            p = QPainter(self)
            p.setCompositionMode(QPainter.CompositionMode_Source)
            p.drawImage(self.rect(), self.image)


STYLE = """
QMainWindow, QDialog { background: #f3f5f9; color: #18263b; }
QWidget { font-family: 'Microsoft YaHei UI'; font-size: 13px; }
QLabel { color: #24344b; }
QFrame#card { background: white; border: 1px solid #e1e6ee; border-radius: 14px; }
QPushButton { padding: 10px 15px; background: white; border: 1px solid #d6dfe9; border-radius: 8px; color: #293b56; }
QPushButton:hover { background: #eaf0fb; border-color: #b4c6e5; }
QPushButton:disabled { color: #a0aabc; background: #edf0f5; }
QPushButton#primary { background: #3859d9; color: white; border-color: #3859d9; }
QPushButton#primary:hover { background: #2f4cc0; }
QPushButton:checked { background: #e8edff; border-color: #3859d9; color: #2945b8; }
QPushButton#disclosure { text-align: left; padding: 10px 12px; background: #eaf0f9; border-color: #dfe6f1; }
QPushButton#quiet { padding: 5px 10px; background: transparent; border-color: transparent; color: #5c6e88; }
QComboBox, QSpinBox { padding: 8px; background: white; border: 1px solid #d6dfe9; border-radius: 7px; }
QCheckBox { spacing: 8px; color: #344661; }
QLabel#muted { color: #748197; }
QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }
QTabBar::tab { padding: 8px 12px; color: #62728b; border-radius: 7px; margin-right: 3px; }
QTabBar::tab:selected { color: #2945b8; background: #e5ebfd; }
QTabBar::tab:hover { background: #eaf0fb; }
"""


class MainWindow(QMainWindow):
    def __init__(self, profile_store=None):
        super().__init__()
        self.setWindowTitle(f"留英文 · 中文字幕局部修补 · {__version__}")
        fit_window(self, (820, 730), (640, 480))
        self.worker = None
        self.window_info = None
        self.packet = None
        self.last_sequence = -1
        self.paused = False
        self.subtitle_style = None
        self.player_identity = None
        self.profile_store = profile_store if profile_store is not None else ProfileStore()
        self.profile_store.load()
        self.overlay = RepairOverlay()
        self.display_count, self.display_since = 0, perf_counter()
        self.presentations = []
        self.last_preview_at = 0.0
        self.last_overlay_rect = None
        self.last_window_check = 0.0
        self.last_overlay_raise = 0.0
        self.selection_open = False
        self.selection_requested_at = None
        self._selection_notice = None
        self._build()
        self.refresh_windows()
        self.hotkey_registered = windows.register_toggle(int(self.winId()))
        self.selection_hotkey_registered = windows.register_selection(int(self.winId()))
        self.selection_shortcut = "Ctrl+Alt+R" if self.selection_hotkey_registered else None
        if not self.selection_hotkey_registered and windows.register_selection_fallback(int(self.winId())):
            self.selection_hotkey_registered = True
            self.selection_shortcut = "Ctrl+Alt+Shift+R"
        select_hint = (f"全屏框选：{self.selection_shortcut}" if self.selection_shortcut else
                       "框选快捷键均被占用，请使用框选按钮")
        if self.selection_shortcut == "Ctrl+Alt+Shift+R":
            select_hint = "Ctrl+Alt+R 已被其他程序占用；已改用 " + select_hint
        toggle_hint = "Ctrl+Alt+H：切换原图" if self.hotkey_registered else "原图快捷键被占用，请使用显示原图按钮"
        self.hotkey_label.setText(select_hint + "  ·  " + toggle_hint)
        self.hotkey_label.setWordWrap(True)
        self.hotkey_events = HotkeyEvents(self)
        self.hotkey_events.activated.connect(self.on_hotkey, Qt.QueuedConnection)
        self.hotkey_filter = HotkeyFilter(int(self.winId()), self.hotkey_events)
        QApplication.instance().installNativeEventFilter(self.hotkey_filter)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.poll)
        self.timer.start(16)

    def _card(self):
        widget = QFrame()
        widget.setObjectName("card")
        return widget

    def _build(self):
        self.comparison = ComparisonWindow(self)
        self.comparison.refresh_requested.connect(lambda: self.refresh_previews(force=True))
        # Keep the result/status widgets in the dedicated comparison window.
        self.previews = self.comparison.previews
        self.metric_labels = self.comparison.metric_labels
        self.export_button = self.comparison.export_button
        self._preview_key = None
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(22, 20, 22, 16)
        layout.setSpacing(14)
        header = QHBoxLayout()
        title_col = QVBoxLayout()
        title_col.setSpacing(4)
        title = QLabel("留英文")
        title.setStyleSheet("font-size:28px;font-weight:700;color:#172843;")
        title_col.addWidget(title)
        subtitle = QLabel("去掉中文干扰，继续在原播放器观看。")
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        title_col.addWidget(subtitle)
        header.addLayout(title_col, 1)
        self.comparison_button = QPushButton("查看效果  A / B / C")
        self.comparison_button.setToolTip("打开独立对比窗口，查看原图、文字轮廓和修补结果。")
        self.comparison_button.clicked.connect(self.open_comparison)
        header.addWidget(self.comparison_button)
        layout.addLayout(header)

        # Advanced options may exceed a small/high-DPI screen; scroll controls
        # instead of squeezing image panels or pushing the footer off screen.
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 6, 0)
        body_layout.setSpacing(12)
        source_card = self._card()
        sc = QVBoxLayout(source_card)
        sc.setContentsMargins(16, 14, 16, 14)
        sc.setSpacing(10)
        heading = QLabel("01  连接视频来源")
        heading.setStyleSheet("font-size:15px;font-weight:600;")
        sc.addWidget(heading)
        row = QHBoxLayout()
        self.window_combo = QComboBox()
        self.window_combo.setMinimumContentsLength(8)
        self.window_combo.setMinimumWidth(80)
        self.window_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        row.addWidget(self.window_combo, 1)
        refresh = QPushButton("刷新")
        refresh.clicked.connect(self.refresh_windows)
        row.addWidget(refresh)
        connect = QPushButton("连接窗口")
        connect.setObjectName("primary")
        connect.clicked.connect(self.connect_window)
        row.addWidget(connect)
        sc.addLayout(row)
        self.source_label = QLabel("支持网页播放器和电脑客户端，请先显示双语字幕。")
        self.source_label.setObjectName("muted")
        self.source_label.setWordWrap(True)
        self.source_label.setMinimumWidth(0)
        sc.addWidget(self.source_label)
        alternatives = QHBoxLayout()
        file = QPushButton("打开本地视频")
        file.clicked.connect(self.open_video)
        alternatives.addWidget(file)
        demo = QPushButton("体验动态样例")
        demo.clicked.connect(self.start_demo)
        alternatives.addWidget(demo)
        alternatives.addStretch()
        sc.addLayout(alternatives)
        body_layout.addWidget(source_card)

        viewing_card = self._card()
        viewing = QVBoxLayout(viewing_card)
        viewing.setContentsMargins(16, 14, 16, 14)
        viewing.setSpacing(10)
        heading = QLabel("02  框选后开始观看")
        heading.setStyleSheet("font-size:15px;font-weight:600;")
        viewing.addWidget(heading)
        controls = QHBoxLayout()
        self.select_button = QPushButton("框选 / 重新校准")
        self.select_button.setObjectName("primary")
        self.select_button.clicked.connect(self.select_region)
        self.select_button.setEnabled(False)
        controls.addWidget(self.select_button, 1)
        self.pause_button = QPushButton("显示原图")
        self.pause_button.clicked.connect(self.toggle_pause)
        controls.addWidget(self.pause_button)
        stop = QPushButton("停止")
        stop.clicked.connect(self.stop_source)
        controls.addWidget(stop)
        viewing.addLayout(controls)
        mode_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("低延迟 · 只框中文", "manual")
        self.mode_combo.addItem("自动区分 · 框选双语", "auto")
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        mode_row.addWidget(self.mode_combo)
        self.continuity_check = QCheckBox("防止切换露字")
        self.continuity_check.setChecked(True)
        self.continuity_check.setToolTip("持续显示处理后的整个中文选区，包括没有字幕时。选区内背景也稍晚更新，快速运动可能看出边界。")
        self.continuity_check.toggled.connect(self.change_presentation)
        mode_row.addWidget(self.continuity_check)
        mode_row.addStretch()
        viewing.addLayout(mode_row)
        self.mode_hint = QLabel("只框中文、不含英文；选区持续显示修补结果，背景略有延迟。")
        self.mode_hint.setWordWrap(True)
        self.mode_hint.setObjectName("muted")
        viewing.addWidget(self.mode_hint)
        self.live_check = QCheckBox("叠加到原播放器")
        self.live_check.setChecked(True)
        self.live_check.stateChanged.connect(lambda _: self.overlay.hide())
        viewing.addWidget(self.live_check)
        body_layout.addWidget(viewing_card)

        self.advanced_toggle = QPushButton("展开高级设置  ·  字体、描边与播放器记忆")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setObjectName("disclosure")
        body_layout.addWidget(self.advanced_toggle)
        self.advanced_panel = self._card()
        advanced = QVBoxLayout(self.advanced_panel)
        advanced.setContentsMargins(16, 14, 16, 14)
        advanced.setSpacing(10)
        tuning = QHBoxLayout()
        tuning.addWidget(QLabel("字体颜色"))
        self.color_combo = QComboBox()
        for label, value in (("白色 / 黄色", "auto"), ("白色", "white"), ("黄色", "yellow")):
            self.color_combo.addItem(label, value)
        self.color_combo.currentIndexChanged.connect(self.change_font_color)
        tuning.addWidget(self.color_combo, 1)
        tuning.addWidget(QLabel("边缘修补"))
        self.padding = QSpinBox()
        self.padding.setRange(0, 5)
        self.padding.setValue(3)
        self.padding.setSuffix(" px")
        self.padding.valueChanged.connect(self.reconfigure)
        tuning.addWidget(self.padding)
        advanced.addLayout(tuning)
        self.style_label = QLabel("使用预设颜色")
        self.style_label.setObjectName("muted")
        advanced.addWidget(self.style_label)
        sampling = QHBoxLayout()
        self.calibrate_button = QPushButton("校准字幕样式")
        self.calibrate_button.setToolTip("暂停在有中文的一帧，框选后点击一个明亮笔画，再预览修补效果。")
        self.calibrate_button.clicked.connect(self.calibrate_subtitle_style)
        self.calibrate_button.setEnabled(False)
        sampling.addWidget(self.calibrate_button)
        self.reset_style_button = QPushButton("恢复默认样式")
        self.reset_style_button.clicked.connect(self.reset_style)
        self.reset_style_button.setEnabled(False)
        sampling.addWidget(self.reset_style_button)
        sampling.addStretch()
        advanced.addLayout(sampling)
        preferences = QHBoxLayout()
        self.save_profile_button = QPushButton("保存播放器设置")
        self.save_profile_button.setToolTip("保存此播放器的范围、字体、边缘及防露字设置。每个播放器保留一份；不保存视频截图。")
        self.save_profile_button.clicked.connect(self.save_profile)
        self.save_profile_button.setEnabled(False)
        preferences.addWidget(self.save_profile_button)
        self.restore_profile_button = QPushButton("恢复播放器设置")
        self.restore_profile_button.clicked.connect(self.restore_profile)
        self.restore_profile_button.setEnabled(False)
        preferences.addWidget(self.restore_profile_button)
        preferences.addStretch()
        advanced.addLayout(preferences)
        self.preference_note = QLabel("校准可减少颜色不匹配；下次连接同一播放器时，可恢复保存的设置并检查范围。")
        self.preference_note.setObjectName("muted")
        self.preference_note.setWordWrap(True)
        advanced.addWidget(self.preference_note)
        self.advanced_panel.hide()
        self.advanced_toggle.toggled.connect(self.toggle_advanced)
        body_layout.addWidget(self.advanced_panel)
        body_layout.addStretch()
        self.controls_scroll.setWidget(body)
        layout.addWidget(self.controls_scroll, 1)

        self.status = QLabel("准备就绪 · 连接播放器，或体验动态样例")
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.status.setStyleSheet("color:#243d6a;font-weight:600;")
        layout.addWidget(self.status)
        self.hotkey_label = QLabel()
        self.hotkey_label.setObjectName("muted")
        self.hotkey_label.setWordWrap(True)
        self.hotkey_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.hotkey_label)
        footer = QHBoxLayout()
        local = QLabel("本机处理  ·  无需上传视频")
        local.setObjectName("muted")
        footer.addWidget(local, 1)
        report = QPushButton("验证报告")
        report.setObjectName("quiet")
        report.clicked.connect(self.open_report)
        footer.addWidget(report)
        about = QPushButton("关于")
        about.setObjectName("quiet")
        about.clicked.connect(self.show_about)
        footer.addWidget(about)
        layout.addLayout(footer)
        self.setCentralWidget(central)

    def toggle_advanced(self, expanded):
        self.advanced_panel.setVisible(expanded)
        self.advanced_toggle.setText("收起高级设置" if expanded else "展开高级设置  ·  字体、描边与播放器记忆")
        if expanded:
            QTimer.singleShot(0, lambda: self.controls_scroll.ensureWidgetVisible(self.advanced_panel))

    def open_comparison(self):
        if self.comparison.isMinimized():
            self.comparison.showNormal()
        else:
            self.comparison.show()
        self.comparison.raise_()
        self.comparison.activateWindow()
        self.refresh_previews(force=True)

    def refresh_previews(self, frame=None, packet=None, *, force=False):
        comparison = self.comparison
        if not comparison.isVisible() or comparison.isMinimized():
            return
        now = perf_counter()
        if not force and now-self.last_preview_at < .10:
            return
        if comparison.frozen:
            if not force:
                return
            frame, packet = comparison.cached_frame, comparison.cached_packet
        elif not self.worker:
            return
        elif frame is None:
            frame, packet = self.worker.snapshot()
        # Snapshot results belong to the current source/epoch. Never show an old
        # stored packet after a player resize or a new selection.
        if packet is not None:
            key = (id(self.worker), packet.epoch, packet.sequence)
            if not force and key == self._preview_key:
                return
            result = packet.result
            for index, preview in enumerate(self.previews):
                if comparison.panels[index].isVisible():
                    image = (result.debug_image() if index == 1 else
                             result.original if index == 0 else result.repaired)
                    preview.set_image(image)
        elif frame is not None:
            key = (id(self.worker), "frame", frame[1])
            if not force and key == self._preview_key:
                return
            self.previews[0].set_image(frame[0])
            self.previews[1].clear("等待框选，随后显示文字轮廓")
            self.previews[2].clear("等待框选，随后显示修补结果")
        else:
            return
        self._preview_key = key
        comparison.cached_frame, comparison.cached_packet = frame, packet
        self.last_preview_at = now

    def refresh_windows(self):
        current = self.window_combo.currentData()
        self.window_combo.clear()
        for info in windows.list_windows():
            self.window_combo.addItem(info.title[:110], info)
        if current:
            for i in range(self.window_combo.count()):
                if self.window_combo.itemData(i).hwnd == current.hwnd:
                    self.window_combo.setCurrentIndex(i)
                    break

    def settings(self):
        return Settings(mode=self.mode_combo.currentData(), font_color=self.color_combo.currentData(),
                        edge_padding=self.padding.value(),
                        style=self.subtitle_style if self.mode_combo.currentData() == "manual" else None)

    def change_mode(self):
        manual = self.mode_combo.currentData() == "manual"
        self.subtitle_style = None
        self.update_style_label()
        self.continuity_check.setEnabled(manual)
        self.update_mode_hint()
        self.overlay.hide()
        self.packet = None
        if self.worker:
            # Never reinterpret an existing bilingual crop as Chinese-only.
            self.worker.configure(None, self.settings())
            self.status.setText("处理方式已切换，请重新框选字幕区域")

    def cover_region(self):
        return self.mode_combo.currentData() == "manual" and self.continuity_check.isChecked()

    def update_mode_hint(self):
        if self.mode_combo.currentData() != "manual":
            text = "框选中英文；后台识别语言，首次定位和切镜需要等待。"
        elif self.cover_region():
            text = "只框中文、不含英文；选区持续显示修补结果，背景略有延迟。"
        else:
            text = "只覆盖已提取的笔画；新字幕出现时可能先短暂露字。"
        self.mode_hint.setText(text)

    def change_presentation(self):
        self.update_mode_hint()
        # Recompose the current result without clearing a valid overlay first.
        self.last_sequence = -1
        self.poll()

    def stop_source(self):
        self.overlay.hide()
        if self.worker:
            self.worker.stop()
            if not self.worker.wait(5000):
                self.status.setText("正在等待识别线程结束，请稍候…")
                return False
            self.worker = None
        self.window_info, self.packet = None, None
        self.player_identity = None
        self.subtitle_style = None
        self.update_style_label()
        self.update_preference_actions(False)
        self.select_button.setEnabled(False)
        self.export_button.setEnabled(False)
        self._preview_key = None
        self.comparison.clear()
        self.status.setText("已停止，画面恢复原样")
        return True

    def start_source(self, **kwargs):
        if not self.stop_source():
            return
        self.window_info = kwargs.get("window")
        if self.window_info:
            self.player_identity = windows.player_identity(self.window_info.hwnd)
        self.preference_note.setText("可恢复已保存的设置，或框选后重新校准样式。" if self.player_identity else
                                    "当前来源可校准样式；保存与恢复设置用于已连接的播放器窗口。")
        self.worker = ProcessingWorker(**kwargs)
        self.worker.message.connect(self.status.setText)
        self.worker.packet_ready.connect(self.on_packet_ready, Qt.QueuedConnection)
        self.worker.finished.connect(self.overlay.hide)
        self.last_sequence = -1
        self.last_preview_at = 0.0
        self.paused = False
        self.pause_button.setText("显示原图")
        self.worker.start()

    @Slot(object)
    def on_packet_ready(self, worker):
        if worker is not self.worker:
            return
        worker.acknowledge_packet_notification()
        self.poll()

    def connect_window(self):
        info = self.window_combo.currentData()
        if info:
            self.start_source(window=info)
            self.source_label.setText(f"当前窗口：{info.title} · 框选完成后，切回播放器即可看到叠加效果")

    def open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开测试视频", "", "视频 (*.mp4 *.mkv *.avi *.mov *.webm)")
        if path:
            self.start_source(video_path=path)
            self.source_label.setText(f"本地对比预览：{Path(path).name}（循环播放，不叠加到其他窗口）")
            self.open_comparison()

    def start_demo(self):
        self.start_source(demo=True)
        if self.worker:
            roi = (.10, .65, .80, .17) if self.mode_combo.currentData() == "manual" else (.08, .60, .84, .36)
            self.worker.configure(roi, self.settings())
            self.source_label.setText("合成动态样例 · 六种场景循环播放 · 不代表真实视频识别准确率")
            self.open_comparison()

    def select_region(self):
        worker = self.worker
        if not worker or self.selection_open:
            if not worker:
                self.status.setText("请先连接正在播放视频的窗口，再框选中文字幕")
            return
        frame, selected_key = worker.selection_snapshot()
        if frame is None:
            self.status.setText("尚未收到播放器画面，请确认窗口可见并等待采集就绪")
            return
        self.overlay.hide()
        old_pause = self.paused
        self.paused = True
        self.selection_open = True
        manual = self.mode_combo.currentData() == "manual"
        dialog = SelectionDialog(frame[0], self, manual=manual)
        try:
            if dialog.exec() == QDialog.Accepted:
                if worker is not self.worker:
                    return
                self.subtitle_style = None
                self.update_style_label()
                active = worker.configure_selection(dialog.canvas.normalized_region(), self.settings(), frame[0], selected_key)
                self.packet = None
                text = ("区域已更新，直接提取中文轮廓…" if manual else "区域已更新，正在识别中文并保护英文…") if active else \
                       "已保存这张画面的选区；返回对应全屏后自动恢复修补"
                self.status.setText(text)
                self.preference_note.setText(text)
        finally:
            self.paused = old_pause
            self.selection_open = False
            dialog.deleteLater()

    def reconfigure(self):
        if self.worker:
            with self.worker.lock:
                roi = self.worker.roi
            self.worker.configure(roi, self.settings())
            self.packet = None
            self.overlay.hide()

    def update_style_label(self):
        calibrated = self.subtitle_style is not None
        self.style_label.setText("已使用采样颜色" if calibrated else "使用预设颜色")
        self.reset_style_button.setEnabled(calibrated)

    def change_font_color(self):
        self.subtitle_style = None
        self.update_style_label()
        self.reconfigure()

    def reset_style(self):
        self.subtitle_style = None
        self.update_style_label()
        self.reconfigure()

    def update_preference_actions(self, has_frame):
        has_roi = has_frame and self.worker is not None and self.worker.roi is not None
        self.calibrate_button.setEnabled(bool(has_roi and self.mode_combo.currentData() == "manual"))
        self.save_profile_button.setEnabled(bool(has_roi and self.player_identity))
        self.restore_profile_button.setEnabled(bool(has_frame and self.player_identity))

    def frame_matches(self, worker, size, epoch=None):
        if self.worker is not worker:
            return False
        frame, _ = worker.snapshot()
        return (frame is not None and frame[0].shape[1::-1] == tuple(size) and
                (epoch is None or epoch == worker.epoch))

    def calibrate_subtitle_style(self):
        worker = self.worker
        if worker is None or self.mode_combo.currentData() != "manual":
            return
        with worker.lock:
            frame, roi, epoch = worker.latest_frame, worker.roi, worker.epoch
        if frame is None or roi is None:
            return
        image = frame[0]
        ih, iw = image.shape[:2]
        rx, ry, rw, rh = roi
        x, y = max(0, round(rx*iw)), max(0, round(ry*ih))
        crop = image[y:min(ih, y+round(rh*ih)), x:min(iw, x+round(rw*iw))]
        if crop.size == 0:
            return
        old_pause = self.paused
        self.paused = True
        self.overlay.hide()
        dialog = None
        try:
            dialog = StyleCalibrationDialog(crop, self, edge_padding=self.padding.value())
            if dialog.exec() == QDialog.Accepted and dialog.style is not None:
                if not self.frame_matches(worker, (iw, ih), epoch):
                    self.preference_note.setText("窗口或选区已变化，请重新框选后校准。")
                    return
                self.subtitle_style = dialog.style
                self.update_style_label()
                self.reconfigure()
                self.preference_note.setText("已应用采样颜色。字幕换色或换字号后，请重新校准；需要复用时点击“保存设置”。")
        finally:
            self.paused = old_pause
            if dialog is not None:
                dialog.deleteLater()

    def save_profile(self):
        worker = self.worker
        if worker is None or not self.player_identity:
            return
        frame, roi, settings = worker.configuration_snapshot()
        if frame is None or roi is None:
            return
        saved = self.profile_store.save(self.player_identity, roi=roi, frame_size=frame[0].shape[1::-1],
            settings={"mode": settings.mode, "font_color": settings.font_color,
                      "edge_padding": settings.edge_padding,
                      "style": settings.style.to_dict() if settings.style else None,
                      "continuity": self.continuity_check.isChecked()})
        self.preference_note.setText("已保存此播放器的设置；恢复时会先显示选区供你检查。" if saved else
                                    self.profile_store.last_error)

    def restore_profile(self):
        worker = self.worker
        if worker is None or not self.player_identity:
            return
        frame, _ = worker.snapshot()
        if frame is None:
            return
        size = frame[0].shape[1::-1]
        profile = self.profile_store.lookup(self.player_identity, size)
        if profile is None:
            self.preference_note.setText(self.profile_store.last_error or "此播放器还没有保存的设置。")
            return
        settings = profile["settings"]
        old_pause = self.paused
        self.paused = True
        self.overlay.hide()
        dialog = None
        try:
            dialog = SelectionDialog(frame[0], self, manual=settings["mode"] == "manual",
                                     initial_roi=profile["roi"])
            if dialog.exec() != QDialog.Accepted:
                return
            if not self.frame_matches(worker, size):
                self.preference_note.setText("窗口尺寸已变化，请重新框选；保存的设置尚未应用。")
                return
            controls = (self.mode_combo, self.color_combo, self.padding, self.continuity_check)
            blocked = [control.blockSignals(True) for control in controls]
            try:
                self.mode_combo.setCurrentIndex(self.mode_combo.findData(settings["mode"]))
                self.color_combo.setCurrentIndex(self.color_combo.findData(settings["font_color"]))
                self.padding.setValue(settings["edge_padding"])
                self.continuity_check.setChecked(settings["continuity"])
                self.subtitle_style = SubtitleStyle.from_dict(settings["style"]) if settings["style"] else None
            finally:
                for control, old in zip(controls, blocked):
                    control.blockSignals(old)
            self.continuity_check.setEnabled(settings["mode"] == "manual")
            self.update_mode_hint()
            self.update_style_label()
            worker.configure(dialog.canvas.normalized_region(), self.settings())
            self.packet = None
            self.last_sequence = -1
            self.preference_note.setText("已恢复设置，并使用刚确认的选区。字幕位置或样式改变后请重新校准。")
        finally:
            self.paused = old_pause
            if dialog is not None:
                dialog.deleteLater()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.setText("恢复修补" if self.paused else "显示原图")
        self.overlay.hide()

    def poll(self):
        if not self.worker:
            return
        self._follow_fullscreen()
        self._finish_selection_request()
        frame, packet = self.worker.snapshot()
        self.select_button.setEnabled(frame is not None and self.worker.isRunning())
        self.update_preference_actions(frame is not None and self.worker.isRunning())
        if not self.worker.isRunning():
            self.overlay.hide()
            return
        if packet is None:
            self.overlay.hide()
            self.refresh_previews(frame, packet)
            return
        age = perf_counter()-packet.captured_at
        if packet.sequence != self.last_sequence:
            self.last_sequence = packet.sequence
            self.packet = packet
            result = packet.result
            self.overlay.set_result(result.repaired, result.mask, cover_region=self.cover_region())
            self.status.setText(result.status + (" · 连续覆盖中文选区" if self.cover_region() else "") +
                                (" · 对比原图中" if self.paused else ""))
            self.metric_labels[0].setText(f"{result.timing['total_ms']:.0f} ms")
            self.metric_labels[1].setText("静止画面" if packet.static_source else f"{age*1000:.0f} ms")
            self.display_count += 1
            elapsed = perf_counter()-self.display_since
            if elapsed >= 1:
                self.metric_labels[2].setText(f"{self.display_count/elapsed:.1f} fps")
                self.display_count, self.display_since = 0, perf_counter()
            self.metric_labels[3].setText(f"{np.count_nonzero(result.mask)/result.mask.size*100:.1f}%")
            self.export_button.setEnabled(True)
            self.presentations.append({**result.timing, "callback_to_present_ms": None if packet.static_source else age*1000,
                                       "presentation": "processed_crop" if self.cover_region() else "glyph_only"})
            self.presentations = self.presentations[-600:]
        # This window has its own visibility and refresh rate. In particular,
        # minimized controls must not freeze a visible comparison window.
        self.refresh_previews(frame, packet)
        newest_capture = frame[1] if frame else packet.captured_at
        if self.worker.source:
            latest_native = self.worker.source.read()
            if latest_native:
                newest_capture = latest_native[2]
        self._update_overlay(packet, newest_capture)

    def _follow_fullscreen(self):
        if not self.window_info or not hasattr(self.worker, "request_window"):
            return
        now = perf_counter()
        if now-self.last_window_check < .25:
            return
        self.last_window_check = now
        current = self.worker.window or self.window_info
        target = windows.follow_fullscreen(current, self.window_info)
        if target is not None and target.hwnd != current.hwnd:
            self.overlay.hide()
            if self.worker.isRunning():
                self.worker.request_window(target)
            else:
                self.worker.restart_window(target)
            self.status.setText("正在跟随播放器切换，自动定位原字幕区域…")

    @Slot(int)
    def on_hotkey(self, action):
        if self.selection_open:
            return
        if action == 1:
            self.toggle_pause()
            return
        if action != 2:
            return
        worker = self.worker
        if worker is None:
            self._selection_error("请先连接播放器；框选快捷键已经收到")
            return
        # The explicit key press can reconnect to a fullscreen surface in the
        # same browser executable, even when its old HWND/process has gone away.
        foreground = windows.foreground_root()
        if (self.window_info and windows.available(foreground) and windows.is_fullscreen(foreground) and
                self.player_identity and windows.player_identity(foreground) == self.player_identity):
            target = windows.window_info(foreground)
            current = worker.window or self.window_info
            if target.hwnd != current.hwnd or not worker.isRunning():
                if worker.isRunning():
                    worker.request_window(target)
                else:
                    worker.restart_window(target)
        self.selection_requested_at = perf_counter()
        self._finish_selection_request()

    def _finish_selection_request(self):
        if self.selection_requested_at is None or self.worker is None:
            return
        worker = self.worker
        frame, key = worker.selection_snapshot()
        pending = getattr(worker, "_requested_window", None) is not None
        target = getattr(worker, "window", None)
        valid = frame is not None and not pending and worker.isRunning()
        if valid and target is not None:
            valid = (key is not None and key[0] == target.hwnd and windows.available(target.hwnd) and
                     max(abs(a-b) for a, b in zip(frame[0].shape[1::-1], windows.bounds(target.hwnd)[2:])) <= 3)
        if valid:
            self.selection_requested_at = None
            QTimer.singleShot(0, self.select_region)
        elif perf_counter()-self.selection_requested_at > 5:
            self.selection_requested_at = None
            self._selection_error("框选快捷键已收到，但没有取得当前全屏画面；请重新连接当前播放窗口")
        else:
            self.status.setText("框选快捷键已收到，正在等待当前播放器画面…")

    def _selection_error(self, text):
        self.status.setText(text)
        if self._selection_notice is None:
            notice = QMessageBox(QMessageBox.Information, "框选尚未就绪", text, QMessageBox.Ok, self)
            notice.setWindowFlag(Qt.WindowStaysOnTopHint)
            notice.setAttribute(Qt.WA_DeleteOnClose)
            notice.finished.connect(lambda _: setattr(self, "_selection_notice", None))
            self._selection_notice = notice
        self._selection_notice.setText(text)
        self._selection_notice.show()
        self._selection_notice.raise_()
        self._selection_notice.activateWindow()

    def _update_overlay(self, packet, newest_capture):
        if (not self.window_info or self.paused or not self.live_check.isChecked() or
            not self.worker.isRunning() or packet.epoch != self.worker.epoch or
            (self.worker.source is not None and self.worker.source.closed) or
            (not self.cover_region() and not np.any(packet.result.mask))):
            self.overlay.hide()
            return
        if not packet.result.safe_to_present:
            self.overlay.hide()
            self.status.setText(packet.result.status)
            return
        if newest_capture-packet.captured_at > .35:
            self.overlay.hide()
            self.status.setText("处理暂时跟不上画面，已恢复原图；跟上后自动恢复修补")
            return
        target = getattr(self.worker, "window", None) or self.window_info
        hwnd = target.hwnd
        if not windows.available(hwnd):
            self.overlay.hide()
            self.status.setText("播放器窗口当前不可见；恢复播放器后显示修补")
            return
        if windows.foreground_root() != hwnd:
            self.overlay.hide()
            self.status.setText("已处理画面，切回连接的播放器后显示修补；全屏可按 Ctrl+Alt+R 框选")
            return
        left, top, width, height = windows.bounds(hwnd)
        fw, fh = packet.frame_size
        # Never stretch an old-size packet over a new window while relocating.
        if abs(width-fw) > 3 or abs(height-fh) > 3:
            self.overlay.hide()
            self.status.setText("窗口尺寸已变化，正在自动定位字幕区域…")
            return
        x,y,w,h = packet.roi
        rect = (left+x, top+y, w,h)
        if rect != self.last_overlay_rect or not self.overlay.isVisible():
            try:
                self.overlay.present_at(rect)
            except OSError:
                self.overlay.hide()
                self.status.setText("无法放置修补层，请重新连接播放器")
                return
            self.last_overlay_rect = rect
            self.last_overlay_raise = perf_counter()
        elif perf_counter()-self.last_overlay_raise > .5:
            # Some fullscreen players reassert TOPMOST after focus returns from
            # the selection dialog. Reassert without resizing or taking focus.
            windows.raise_overlay(int(self.overlay.winId()))
            self.last_overlay_raise = perf_counter()

    def export_snapshot(self):
        packet = self.comparison.cached_packet if self.comparison.frozen else self.packet
        if not packet:
            return
        # Capture the selected result before the native folder dialog's nested
        # event loop allows newer processing packets to arrive.
        result = packet.result
        directory = QFileDialog.getExistingDirectory(self.comparison, "选择对比快照保存位置")
        if not directory:
            return
        from datetime import datetime
        dest = Path(directory) / ("subtitle-compare-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        dest.mkdir(parents=True, exist_ok=True)
        for name, image in (("original",result.original),("mask",result.mask),("repaired",result.repaired),("contours",result.debug_image())):
            cv2.imencode(".png",image)[1].tofile(dest/f"{name}.png")
        (dest/"metrics.json").write_text(json.dumps(self.presentations,ensure_ascii=False,indent=2),encoding="utf-8")
        self.status.setText(f"已保存对比与耗时记录：{dest}")

    def open_report(self):
        report = ROOT / "outputs" / "validation" / "report.html"
        if report.exists():
            os.startfile(report)
        else:
            self.show_document("验证说明", ROOT / "docs" / "VALIDATION.md",
                               "此版本未附生成的动态报告。源码版可运行 validate.cmd 生成；合成测量不代表真实视频兼容性验收。")

    def show_document(self, title, path, fallback):
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(850, 620)
        layout = QVBoxLayout(dialog)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.document().setBaseUrl(QUrl.fromLocalFile(str(path.parent) + "/"))
        browser.setMarkdown(path.read_text(encoding="utf-8") if path.is_file() else fallback)
        layout.addWidget(browser)
        close = QPushButton("关闭")
        close.clicked.connect(dialog.accept)
        layout.addWidget(close)
        try:
            dialog.exec()
        finally:
            dialog.deleteLater()

    def show_about(self):
        self.show_document(f"留英文 {__version__} · 开源许可", ROOT / "THIRD_PARTY.md",
            f"留英文 {__version__}，项目代码采用 MIT 许可证。界面使用 PySide6 / Qt（LGPLv3）；第三方组件遵循各自许可证。")

    def nativeEvent(self, event_type, message):
        return super().nativeEvent(event_type, message)

    def closeEvent(self, event):
        if self.stop_source():
            windows.unregister_toggle(int(self.winId()))
            windows.unregister_selection(int(self.winId()))
            QApplication.instance().removeNativeEventFilter(self.hotkey_filter)
            self.overlay.close()
            self.comparison.close()
            self.comparison.deleteLater()
            event.accept()
        else:
            event.ignore()


def run(demo=False):
    app = QApplication.instance() or QApplication([])
    family = "Microsoft YaHei UI"
    if family not in QFontDatabase.families():
        bundled = Path(__file__).resolve().parent / "assets" / "fonts" / "NotoSansSC.ttf"
        font_id = QFontDatabase.addApplicationFont(str(bundled))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            family = families[0]
    app.setStyleSheet(STYLE.replace("Microsoft YaHei UI", family))
    app.setFont(QFont(family, 10))
    window = MainWindow()
    window.show()
    if demo:
        QTimer.singleShot(400, window.start_demo)
    return app.exec()
