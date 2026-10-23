import pytest
from subtitle_eraser import windows


@pytest.mark.parametrize('foreground,pid,full,available,expected', [
    (20, 123, True, True, 20),   # New fullscreen HWND in the selected process.
    (20, 456, True, True, None), # Another app must never be captured automatically.
    (20, 123, False, True, None),# Another normal window in the same process.
    (20, 123, True, False, None),
    (10, 123, True, True, None), # Already capturing the foreground HWND.
    (0, 123, True, True, None),
])
def test_follow_only_same_process_fullscreen(monkeypatch, foreground, pid, full, available, expected):
    monkeypatch.setattr(windows, 'foreground_root', lambda: foreground)
    monkeypatch.setattr(windows, 'available', lambda _: available)
    monkeypatch.setattr(windows, 'process_id', lambda h: 123 if h == 10 else pid)
    monkeypatch.setattr(windows, 'is_fullscreen', lambda _: full)
    monkeypatch.setattr(windows.user32, 'GetWindowTextLengthW', lambda _: 0)
    monkeypatch.setattr(windows.user32, 'GetWindowTextW', lambda *_: 0)
    original = windows.WindowInfo(10, 'selected')
    result = windows.follow_fullscreen(original, original)
    assert (result.hwnd if result else None) == expected


def test_return_to_original_window_after_fullscreen(monkeypatch):
    original = windows.WindowInfo(10, 'selected')
    monkeypatch.setattr(windows, 'foreground_root', lambda: 10)
    monkeypatch.setattr(windows, 'available', lambda _: True)
    assert windows.follow_fullscreen(windows.WindowInfo(20, 'fullscreen'), original) == original


def test_follow_replacement_when_original_native_handle_was_destroyed(monkeypatch):
    original = windows.WindowInfo(10, 'selected', 123)
    monkeypatch.setattr(windows, 'foreground_root', lambda: 20)
    monkeypatch.setattr(windows, 'available', lambda _: True)
    monkeypatch.setattr(windows, 'process_id', lambda h: 0 if h == 10 else 123)
    monkeypatch.setattr(windows, 'is_fullscreen', lambda _: True)
    monkeypatch.setattr(windows.user32, 'GetWindowTextLengthW', lambda _: 0)
    monkeypatch.setattr(windows.user32, 'GetWindowTextW', lambda *_: 0)
    result = windows.follow_fullscreen(original, original)
    assert result.hwnd == 20 and result.pid == 123
