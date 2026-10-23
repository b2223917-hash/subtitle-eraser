from __future__ import annotations

import inspect
from threading import Condition, Lock
from time import perf_counter

import numpy as np
from windows_capture import WindowsCapture


class WindowCapture:
    """WGC source, latest-frame mailbox; no unbounded queue or screen scraping."""
    def __init__(self, hwnd: int, title: str):
        self.lock = Lock()
        self.frame_available = Condition(self.lock)
        self.latest = None
        self.sequence = 0
        self.closed = False
        self.error = ""
        params = dict(cursor_capture=False, draw_border=True)
        signature = inspect.signature(WindowsCapture)
        if "window_hwnd" in signature.parameters:
            params["window_hwnd"] = hwnd
        else:
            # Older builds match titles; reject ambiguity instead of capturing another app.
            from .windows import list_windows
            if sum(title in w.title for w in list_windows()) != 1:
                raise RuntimeError("窗口标题不唯一，请关闭同名窗口后重试")
            params["window_name"] = title
        self.capture = WindowsCapture(**params)
        self.capture.frame_handler = self._on_frame
        self.capture.closed_handler = self._on_closed
        self.control = None

    def _on_frame(self, frame, control):
        if self.closed:
            control.stop()
            return
        try:
            captured_at = perf_counter()
            # Native buffers can be reused immediately after the callback returns.
            image = np.ascontiguousarray(frame.frame_buffer[:, :, :3]).copy()
            with self.frame_available:
                if self.closed:
                    return
                self.sequence += 1
                self.latest = (self.sequence, image, captured_at)
                self.frame_available.notify_all()
        except Exception as exc:
            with self.frame_available:
                self.error = str(exc)
                self.closed = True
                self.frame_available.notify_all()
            control.stop()

    def _on_closed(self):
        with self.frame_available:
            self.closed = True
            self.frame_available.notify_all()

    def start(self):
        self.control = self.capture.start_free_threaded()

    def read(self):
        with self.lock:
            return self.latest

    def wait_for_frame(self, previous_sequence, timeout=.04):
        """Wake on a newer capture; on timeout return the static latest frame."""
        with self.frame_available:
            self.frame_available.wait_for(
                lambda: self.closed or (self.latest is not None and
                                       self.latest[0] != previous_sequence),
                timeout=max(0.0, timeout),
            )
            return self.latest

    def stop(self):
        self._on_closed()
        if self.control is not None:
            self.control.stop()
            self.control = None
