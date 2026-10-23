import ctypes
from ctypes import wintypes
from types import SimpleNamespace
import pytest
from PySide6.QtWidgets import QApplication
from subtitle_eraser import windows
from subtitle_eraser.hotkeys import HotkeyFilter
from subtitle_eraser.ui import MainWindow


def test_global_native_filter_dispatches_only_our_registered_actions():
    received = []
    events = SimpleNamespace(activated=SimpleNamespace(emit=received.append))
    event_filter = HotkeyFilter(123, events)
    msg = wintypes.MSG()
    msg.hWnd, msg.message, msg.wParam = 123, 0x0312, 2
    assert event_filter.nativeEventFilter(b'windows_dispatcher_MSG', ctypes.addressof(msg)) == (True, 0)
    assert received == [2]
    msg.hWnd = 456
    assert event_filter.nativeEventFilter(b'windows_dispatcher_MSG', ctypes.addressof(msg)) == (False, 0)
    msg.hWnd, msg.wParam = 123, 99
    assert event_filter.nativeEventFilter(b'windows_dispatcher_MSG', ctypes.addressof(msg)) == (False, 0)
    assert received == [2]


def test_occupied_primary_shortcut_uses_and_displays_fallback(monkeypatch, tmp_path):
    from subtitle_eraser.profiles import ProfileStore
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(windows, 'register_selection', lambda _: False)
    monkeypatch.setattr(windows, 'register_selection_fallback', lambda _: True)
    monkeypatch.setattr(windows, 'unregister_selection', lambda _: None)
    window = MainWindow(profile_store=ProfileStore(tmp_path/'profiles.json'))
    try:
        assert window.selection_shortcut == 'Ctrl+Alt+Shift+R'
        assert '全屏框选：Ctrl+Alt+Shift+R' in window.hotkey_label.text()
        monkeypatch.setattr(window, '_selection_error', window.status.setText)
        window.on_hotkey(2)
        assert '请先连接播放器' in window.status.text()
        assert '已经收到' in window.status.text()
    finally:
        window.close()
        window.deleteLater()
