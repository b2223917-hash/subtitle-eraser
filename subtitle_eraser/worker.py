from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event, Lock
from time import perf_counter

import cv2
from PySide6.QtCore import QThread, Signal

from .pipeline import SubtitleProcessor, Settings, AsyncOcrDetector
from .relocation import relocate_region
from .row_extent import fit_manual_row


@dataclass
class Packet:
    sequence: int
    result: object
    captured_at: float
    completed_at: float
    frame_size: tuple[int, int]
    roi: tuple[int, int, int, int]
    epoch: int
    static_source: bool = False


class ProcessingWorker(QThread):
    message = Signal(str)
    packet_ready = Signal(object)

    def __init__(self, *, window=None, video_path=None, demo=False):
        super().__init__()
        self.window, self.video_path, self.demo = window, video_path, demo
        self.stop_event = Event()
        self.lock = Lock()
        self.roi = None
        self._roi_frame_size = None
        self.settings = Settings()
        self.epoch = 0
        self.latest_frame = None
        self.latest_packet = None
        self._notification_pending = False
        self.source = None
        self._requested_window = None
        self._relocation = None
        self._layouts = {}
        self._force_relocation = False
        self._style_roi_height = None
        self._reference_roi_height = None
        self._captured_key = None

    def configure(self, roi, settings):
        with self.lock:
            if roi is None:
                self._layouts.clear()
            self._relocation = None
            if roi != self.roi or self._reference_roi_height is None:
                self._reference_roi_height = (roi[3]*self.latest_frame[0].shape[0]
                    if roi is not None and self.latest_frame is not None else None)
            if settings.style != self.settings.style or self._style_roi_height is None:
                self._style_roi_height = (roi[3]*self.latest_frame[0].shape[0]
                    if roi is not None and self.latest_frame is not None and settings.style is not None else None)
            self.roi = roi
            # Bind a selection to the frame it was made against. An epoch reset
            # alone must not rebind an old selection to a newly resized frame.
            self._roi_frame_size = (self.latest_frame[0].shape[:2]
                                    if roi is not None and self.latest_frame is not None else None)
            self.settings = settings
            self.epoch += 1
            self.latest_packet = None
            if roi is not None and self.latest_frame is not None:
                # Persist an explicit selection immediately. A fullscreen player
                # may resize on dialog focus before the next processing frame.
                self._layouts[self._current_frame_key()] = roi

    def _current_frame_key(self):
        if self.latest_frame is None:
            return None
        shape = self.latest_frame[0].shape[:2]
        if self._captured_key is not None and self._captured_key[1] == tuple(shape):
            return self._captured_key
        return self._layout_key(shape)

    def selection_snapshot(self):
        with self.lock:
            return self.latest_frame, self._current_frame_key()

    def configure_selection(self, roi, settings, selected_image, selected_key):
        """Bind confirmation to the screenshot shown, even if focus exits fullscreen."""
        with self.lock:
            self._layouts[selected_key] = roi
            self.settings = settings
            self._reference_roi_height = roi[3]*selected_image.shape[0]
            self._style_roi_height = self._reference_roi_height if settings.style else None
            self.epoch += 1
            self.latest_packet = None
            if self._current_frame_key() == selected_key:
                self.roi, self._roi_frame_size, self._relocation = roi, selected_image.shape[:2], None
                return True
            self.roi, self._roi_frame_size = None, None
            self._relocation = dict(image=selected_image, roi=roi, started=perf_counter(), attempted=0., warned=False)
            return False

    def request_window(self, window):
        with self.lock:
            self._requested_window = window
            self.latest_packet = None
            self.epoch += 1

    def restart_window(self, window):
        """Restart a finished capture without losing this connection's layouts."""
        if self.isRunning():
            return False
        with self.lock:
            self.window = window
            self._requested_window = None
            self._force_relocation = True
            self.latest_packet = None
            self.source = None
            self._notification_pending = False
            self.stop_event.clear()
            self.epoch += 1
        self.start()
        return True

    def _layout_key(self, shape):
        return (self.window.hwnd if self.window else 0, tuple(shape))

    def _accept_frame(self, image, captured_at):
        """Invalidate stale output atomically; registration runs outside the lock."""
        with self.lock:
            if self._requested_window is not None:
                return None, self.settings, self.epoch
            changed = self._force_relocation or (self.roi is not None and
                self._roi_frame_size is not None and image.shape[:2] != self._roi_frame_size) or (
                self._relocation is not None and self.latest_frame is not None and
                image.shape[:2] != self.latest_frame[0].shape[:2])
            self._force_relocation = False
            if changed:
                if self.roi is not None and self.latest_frame is not None:
                    self._relocation = dict(image=self.latest_frame[0], roi=self.roi,
                                            started=perf_counter(), attempted=0., warned=False)
                elif self._relocation is not None:
                    self._relocation = {**self._relocation, "started": perf_counter(), "attempted": 0., "warned": False}
                self.roi, self.latest_packet, self._roi_frame_size = None, None, None
                self.epoch += 1
            self.latest_frame = image, captured_at
            self._captured_key = self._layout_key(image.shape[:2])
            job, epoch = self._relocation, self.epoch
            remembered = self._layouts.get(self._layout_key(image.shape[:2])) if job else None
        if changed:
            self.message.emit("窗口尺寸已变化，正在自动定位字幕区域…")
        if job is not None and not self.stop_event.is_set():
            now = perf_counter()
            found = remembered
            if found is None and now-job["attempted"] >= .5 and now-job["started"] < 6:
                job["attempted"] = now
                found = relocate_region(job["image"], image, job["roi"])
            with self.lock:
                if epoch == self.epoch and job is self._relocation and found is not None:
                    self.roi, self._relocation = found, None
                    self._roi_frame_size = image.shape[:2]
                    self.epoch += 1
                    restored = True
                else:
                    restored = False
            if restored:
                self.message.emit("已跟随窗口变化，恢复字幕修补")
            elif now-job["started"] >= 6 and not job["warned"]:
                job["warned"] = True
                self.message.emit("未能可靠定位新布局，请在当前画面重新框选一次；切回已记住的布局会自动恢复")
        with self.lock:
            row_roi, row_settings, row_epoch = self.roi, self._effective_settings(), self.epoch
        fitted = fit_manual_row(image, row_roi, row_settings) if row_roi is not None else None
        with self.lock:
            if self._requested_window is not None:
                return None, self.settings, self.epoch
            if row_epoch == self.epoch and fitted is not None and fitted != self.roi:
                self.roi = fitted
                self.latest_packet = None
                self.epoch += 1
            if self.roi is not None:
                self._roi_frame_size = image.shape[:2]
                if self._reference_roi_height is None:
                    self._reference_roi_height = self.roi[3]*image.shape[0]
                self._layouts[self._layout_key(image.shape[:2])] = self.roi
                # Keep only a few layouts during this connection; never screenshots on disk.
                while len(self._layouts) > 6:
                    del self._layouts[next(iter(self._layouts))]
            return self.roi, self._effective_settings(), self.epoch

    def _effective_settings(self):
        # Caller holds self.lock. The UI's calibrated style remains the reference
        # size; derive a current-size copy for both processing and profile saves.
        settings = self.settings
        if self.roi is not None and self.latest_frame is not None and self._reference_roi_height:
            scale = self.roi[3]*self.latest_frame[0].shape[0]/self._reference_roi_height
            settings = replace(settings, processing_scale=max(.3, min(4.5, scale)))
        if (self.roi is not None and self.latest_frame is not None and settings.style is not None and
                settings.style.glyph_height is not None and self._style_roi_height):
            scale = self.roi[3]*self.latest_frame[0].shape[0]/self._style_roi_height
            height = max(8, min(120, round(settings.style.glyph_height*scale)))
            settings = replace(settings, style=replace(settings.style, glyph_height=height))
        return settings

    def configuration_snapshot(self):
        with self.lock:
            return self.latest_frame, self.roi, self._effective_settings()

    def snapshot(self):
        with self.lock:
            return self.latest_frame, self.latest_packet

    def stop(self):
        self.stop_event.set()

    def publish_packet(self, packet):
        """Publish only the latest result and keep at most one queued UI wakeup."""
        with self.lock:
            if self.stop_event.is_set() or packet.epoch != self.epoch:
                return False
            self.latest_packet = packet
            notify = not self._notification_pending
            self._notification_pending = True
        # The UI connects with QueuedConnection; never emit while holding the lock.
        if notify:
            self.packet_ready.emit(self)
        return True

    def acknowledge_packet_notification(self):
        # Acknowledge before taking the snapshot: an arriving frame can then queue
        # another wakeup, so it cannot be stranded behind this acknowledgement.
        with self.lock:
            self._notification_pending = False

    def run(self):
        video = None
        detector = None
        try:
            if self.window:
                from .capture import WindowCapture
                self.source = WindowCapture(self.window.hwnd, self.window.title)
                self.source.start()
            elif self.video_path:
                video = cv2.VideoCapture(self.video_path)
                if not video.isOpened():
                    raise RuntimeError("无法打开视频文件")
            detector = AsyncOcrDetector()
            processor = SubtitleProcessor(detector)
            self.message.emit("采集已就绪，请按所选模式框选字幕区域")
            sequence, previous, local_epoch = 0, -1, -1
            fps = min(60, max(1, video.get(cv2.CAP_PROP_FPS) or 24)) if video else 24
            next_frame = perf_counter()
            started = next_frame
            last_processed = 0.0
            closed_since = None
            size_candidate, size_since, restart_at, restart_count = None, 0., 0., 0
            while not self.stop_event.is_set():
                if self.window:
                    with self.lock:
                        requested, self._requested_window = self._requested_window, None
                        if requested is not None:
                            self.window = requested
                            self._force_relocation = True
                    if requested is not None:
                        self.source.stop()
                        self.source = WindowCapture(requested.hwnd, requested.title)
                        self.source.start()
                        previous, started, closed_since = -1, perf_counter(), None
                        size_candidate, restart_count = None, 0
                    with self.lock:
                        has_roi = self.roi is not None or self._relocation is not None
                    until_static_refresh = max(0.0, .10-(perf_counter()-last_processed))
                    timeout = min(.04, until_static_refresh) if has_roi and last_processed else .04
                    frame_data = self.source.wait_for_frame(previous, timeout)
                    if self.stop_event.is_set():
                        break
                    if self.source.closed:
                        closed_since = closed_since or perf_counter()
                        if perf_counter()-closed_since > 3:
                            raise RuntimeError(self.source.error or "目标窗口已关闭，或采集已停止；请重新连接播放器")
                        self.stop_event.wait(.04)
                        continue
                    closed_since = None
                    # Some WGC/driver combinations keep the original frame pool
                    # after a window resize. Do not wait forever for new-sized
                    # packets, or feed a clipped old-size image into relocation.
                    if frame_data is not None:
                        from . import windows
                        if windows.available(self.window.hwnd):
                            native_size = windows.bounds(self.window.hwnd)[2:]
                            captured_size = frame_data[1].shape[1::-1]
                            if max(abs(a-b) for a, b in zip(native_size, captured_size)) > 3:
                                now = perf_counter()
                                if size_candidate != native_size:
                                    size_candidate, size_since, restart_count = native_size, now, 0
                                if now-size_since > .3 and now-restart_at > 1:
                                    if restart_count >= 3:
                                        raise RuntimeError("采集尺寸无法跟随播放器，请重新连接当前全屏窗口")
                                    self.source.stop()
                                    self.source = WindowCapture(self.window.hwnd, self.window.title)
                                    self.source.start()
                                    previous, started, restart_at = -1, now, now
                                    restart_count += 1
                                    with self.lock:
                                        self._force_relocation = True
                                        self.latest_packet = None
                                        self.epoch += 1
                                self.stop_event.wait(.01)
                                continue
                            size_candidate, restart_count = None, 0
                    if frame_data is None or (frame_data[0] == previous and
                                             (not has_roi or perf_counter()-last_processed < .10)):
                        if frame_data is None and perf_counter()-started > 8:
                            raise RuntimeError("未收到窗口画面；视频可能受保护，或窗口不可捕获")
                        continue
                    static_source = frame_data[0] == previous
                    previous, image, captured_at = frame_data
                    sequence += 1
                    last_processed = perf_counter()
                else:
                    static_source = False
                    remaining = next_frame - perf_counter()
                    if remaining > 0:
                        self.stop_event.wait(min(remaining, .04))
                        continue
                    captured_at = perf_counter()
                    if video:
                        # Skip late frames rather than accumulating video delay.
                        if captured_at-next_frame > 1/fps:
                            skip = min(int((captured_at-next_frame)*fps), 120)
                            for _ in range(skip):
                                video.grab()
                        ok, image = video.read()
                        if not ok:
                            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, image = video.read()
                            processor.reset()
                        if not ok:
                            raise RuntimeError("视频没有可读取的画面")
                    else:
                        from .fixtures import sample_frame
                        image, _, _, _ = sample_frame(int((captured_at-started)*24))
                    sequence += 1
                    next_frame = captured_at + 1/fps
                roi, settings, epoch = self._accept_frame(image, captured_at)
                if epoch != local_epoch:
                    processor.reset()
                    processor.settings = settings
                    local_epoch = epoch
                if roi is None:
                    continue
                ih, iw = image.shape[:2]
                rx, ry, rw, rh = roi
                x, y = max(0, round(rx*iw)), max(0, round(ry*ih))
                w, h = min(iw-x, round(rw*iw)), min(ih-y, round(rh*ih))
                if w < 20 or h < 12:
                    continue
                result = processor.process(image[y:y+h, x:x+w])
                last_processed = perf_counter()
                packet = Packet(sequence, result, captured_at, last_processed, (iw, ih), (x,y,w,h), epoch, static_source)
                self.publish_packet(packet)
        except Exception as exc:
            self.message.emit(f"处理已停止：{exc}")
        finally:
            if detector:
                detector.close()
            if self.source:
                self.source.stop()
            if video:
                video.release()
