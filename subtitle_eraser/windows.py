"""Small Win32 boundary; all coordinates in this module are physical pixels."""
from __future__ import annotations

import ctypes as C
from ctypes import wintypes as W
from dataclasses import dataclass
import os

user32 = C.WinDLL("user32", use_last_error=True)
dwmapi = C.WinDLL("dwmapi", use_last_error=True)
user32.GetForegroundWindow.restype = W.HWND
user32.GetAncestor.argtypes = [W.HWND, W.UINT]
user32.GetAncestor.restype = W.HWND
user32.GetWindowRect.argtypes = [W.HWND, C.POINTER(W.RECT)]
user32.GetWindowThreadProcessId.argtypes = [W.HWND, C.POINTER(W.DWORD)]
user32.IsWindow.argtypes = [W.HWND]
user32.IsIconic.argtypes = [W.HWND]
user32.IsZoomed.argtypes = [W.HWND]
user32.IsWindowVisible.argtypes = [W.HWND]
user32.GetWindowTextLengthW.argtypes = [W.HWND]
user32.GetWindowTextW.argtypes = [W.HWND, W.LPWSTR, C.c_int]
user32.SetWindowDisplayAffinity.argtypes = [W.HWND, W.DWORD]
user32.SetWindowPos.argtypes = [W.HWND, W.HWND, C.c_int, C.c_int, C.c_int, C.c_int, W.UINT]
user32.RegisterHotKey.argtypes = [W.HWND, C.c_int, W.UINT, W.UINT]
user32.UnregisterHotKey.argtypes = [W.HWND, C.c_int]
user32.MonitorFromWindow.argtypes = [W.HWND, W.DWORD]
user32.MonitorFromWindow.restype = W.HANDLE
user32.GetMonitorInfoW.argtypes = [W.HANDLE, C.c_void_p]
dwmapi.DwmGetWindowAttribute.argtypes = [W.HWND, W.DWORD, C.c_void_p, W.DWORD]


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str
    pid: int = 0


def list_windows() -> list[WindowInfo]:
    results = []
    callback_type = C.WINFUNCTYPE(W.BOOL, W.HWND, W.LPARAM)

    def collect(hwnd, _):
        if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
            return True
        pid = W.DWORD()
        user32.GetWindowThreadProcessId(hwnd, C.byref(pid))
        if pid.value == os.getpid():
            return True
        size = user32.GetWindowTextLengthW(hwnd)
        if size:
            title = C.create_unicode_buffer(size + 1)
            user32.GetWindowTextW(hwnd, title, size + 1)
            rect = bounds(hwnd)
            if rect[2] >= 100 and rect[3] >= 80 and title.value not in ("Program Manager",):
                results.append(WindowInfo(int(hwnd), title.value, pid.value))
        return True

    callback = callback_type(collect)
    user32.EnumWindows(callback, 0)
    return sorted(results, key=lambda w: w.title.casefold())


def bounds(hwnd: int) -> tuple[int, int, int, int]:
    rect = W.RECT()
    # WGC window frames align with DWM's extended frame, excluding invisible borders.
    if dwmapi.DwmGetWindowAttribute(hwnd, 9, C.byref(rect), C.sizeof(rect)) != 0:
        if not user32.GetWindowRect(hwnd, C.byref(rect)):
            raise OSError("目标窗口已关闭")
    return rect.left, rect.top, rect.right-rect.left, rect.bottom-rect.top


def available(hwnd: int) -> bool:
    return bool(user32.IsWindow(hwnd) and user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd))


def foreground_root() -> int:
    handle = user32.GetForegroundWindow()
    return int(user32.GetAncestor(handle, 2) or handle or 0)


def exclude_from_capture(hwnd: int) -> bool:
    return bool(user32.SetWindowDisplayAffinity(hwnd, 0x11))


def position_overlay(hwnd: int, rect: tuple[int, int, int, int]):
    # HWND_TOPMOST; SWP_NOACTIVATE. Use Win32 to avoid mixed-DPI Qt rounding errors.
    if not user32.SetWindowPos(hwnd, C.c_void_p(-1), *rect, 0x0010):
        raise C.WinError(C.get_last_error())


def process_id(hwnd: int) -> int:
    pid = W.DWORD()
    user32.GetWindowThreadProcessId(hwnd, C.byref(pid))
    return pid.value


def is_fullscreen(hwnd: int) -> bool:
    class MonitorInfo(C.Structure):
        _fields_ = [("size", W.DWORD), ("monitor", W.RECT),
                    ("work", W.RECT), ("flags", W.DWORD)]
    info = MonitorInfo()
    info.size = C.sizeof(info)
    monitor = user32.MonitorFromWindow(hwnd, 2)
    if not monitor or not user32.GetMonitorInfoW(monitor, C.byref(info)):
        return False
    x, y, width, height = bounds(hwnd)
    r = info.monitor
    fullscreen = all(abs(a-b) <= 4 for a, b in zip(
        (x, y, x+width, y+height), (r.left, r.top, r.right, r.bottom)))
    if fullscreen:
        return True
    r = info.work
    return bool(user32.IsZoomed(hwnd)) and all(abs(a-b) <= 4 for a, b in zip(
        (x, y, x+width, y+height), (r.left, r.top, r.right, r.bottom)))


def follow_fullscreen(current: WindowInfo, original: WindowInfo) -> WindowInfo | None:
    """Follow a foreground fullscreen surface in the SAME process only.

    Ordinary sibling windows/tabs and unrelated foreground apps never qualify.
    Return to the explicitly selected original window when it regains focus.
    """
    foreground = foreground_root()
    if not foreground or foreground == current.hwnd or not available(foreground):
        return None
    if foreground == original.hwnd:
        return original
    pid = process_id(current.hwnd) or current.pid or process_id(original.hwnd) or original.pid
    if not pid or process_id(foreground) != pid or not is_fullscreen(foreground):
        return None
    title = C.create_unicode_buffer(user32.GetWindowTextLengthW(foreground)+1)
    user32.GetWindowTextW(foreground, title, len(title))
    return WindowInfo(foreground, title.value, pid)


def window_info(hwnd: int) -> WindowInfo:
    title = C.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd)+1)
    user32.GetWindowTextW(hwnd, title, len(title))
    return WindowInfo(hwnd, title.value, process_id(hwnd))


def register_toggle(hwnd: int) -> bool:
    return bool(user32.RegisterHotKey(hwnd, 1, 0x4000 | 0x0002 | 0x0001, ord("H")))


def register_selection(hwnd: int) -> bool:
    return bool(user32.RegisterHotKey(hwnd, 2, 0x4000 | 0x0002 | 0x0001, ord("R")))


def register_selection_fallback(hwnd: int) -> bool:
    return bool(user32.RegisterHotKey(hwnd, 2, 0x4000 | 0x0002 | 0x0001 | 0x0004, ord("R")))


def unregister_selection(hwnd: int):
    user32.UnregisterHotKey(hwnd, 2)


def raise_overlay(hwnd: int):
    user32.SetWindowPos(hwnd, C.c_void_p(-1), 0, 0, 0, 0, 0x0010 | 0x0001 | 0x0002)


def unregister_toggle(hwnd: int):
    user32.UnregisterHotKey(hwnd, 1)


def player_identity(hwnd: int) -> str | None:
    """Read executable identity for profile matching; never uses a window title."""
    if not hwnd or not user32.IsWindow(hwnd):
        return None
    pid = W.DWORD()
    user32.GetWindowThreadProcessId(hwnd, C.byref(pid))
    if not pid.value:
        return None
    # psutil already ships with the application. Protected processes may refuse
    # executable queries; an unavailable identity simply disables profile reuse.
    import psutil
    try:
        process = psutil.Process(pid.value)
        return process.exe() or None
    except (psutil.Error, OSError):
        return None
