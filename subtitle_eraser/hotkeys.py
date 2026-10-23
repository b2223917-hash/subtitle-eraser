"""Receive registered Windows hotkeys at the application dispatcher.

Only enqueue actions here: opening a modal dialog inside a native message
callback can interrupt the key event dispatch while modifiers are still down.
"""
import ctypes
from ctypes import wintypes
from PySide6.QtCore import QAbstractNativeEventFilter, QObject, Signal


class HotkeyEvents(QObject):
    activated = Signal(int)


class HotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, hwnd, events):
        super().__init__()
        self.hwnd, self.events = hwnd, events

    def nativeEventFilter(self, event_type, message):
        msg = wintypes.MSG.from_address(int(message))
        if msg.message == 0x0312 and int(msg.hWnd or 0) == self.hwnd and int(msg.wParam) in (1, 2):
            self.events.activated.emit(int(msg.wParam))
            return True, 0
        return False, 0
