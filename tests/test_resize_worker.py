"""Real worker sequence tests with synthetic frames only; no desktop capture."""
from threading import Lock

import numpy as np
from PySide6.QtCore import Qt

from subtitle_eraser import fixtures
from subtitle_eraser.pipeline import Settings
from subtitle_eraser.worker import ProcessingWorker


class ObservedLock:
    """Record the state visible to readers at every release of the real lock."""
    def __init__(self, worker):
        self.worker = worker
        self.lock = Lock()
        self.resized_states = []

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, *exc):
        worker = self.worker
        if worker.latest_frame is not None and worker.latest_frame[0].shape[:2] == (120, 240):
            self.resized_states.append((worker.roi, worker.latest_packet, worker.epoch))
        self.lock.release()


def test_resize_atomically_invalidates_roi_and_old_epoch_packets(monkeypatch):
    worker = ProcessingWorker(demo=True)
    observed = ObservedLock(worker)
    worker.lock = observed
    small = np.full((80, 160, 3), 30, np.uint8)
    large = np.full((120, 240, 3), 30, np.uint8)
    reads = []

    def generated(_):
        reads.append(1)
        # The first processed frame belongs to the selected size; the next one
        # simulates a player resize between captures.
        return (small if len(reads) == 1 else large), None, None, []

    monkeypatch.setattr(fixtures, "sample_frame", generated)
    worker.configure((0, 0, 1, 1), Settings(mode="manual"))
    old_epoch = worker.epoch
    packets, errors, stale_results = [], [], []

    def ready(_):
        frame, packet = worker.snapshot()
        packets.append(packet)
        worker.acknowledge_packet_notification()

    def message(text):
        if "尺寸已变化" in text:
            stale_results.append(worker.publish_packet(packets[0]))
            worker.stop()
        elif "已停止" in text:
            errors.append(text)
            worker.stop()

    worker.packet_ready.connect(ready, Qt.ConnectionType.DirectConnection)
    worker.message.connect(message, Qt.ConnectionType.DirectConnection)
    worker.start()
    try:
        assert worker.wait(4000)
    finally:
        worker.stop()
        worker.wait(4000)
    assert not errors
    assert len(packets) == 1 and packets[0].epoch == old_epoch
    assert stale_results == [False]
    assert observed.resized_states
    assert all(roi is None and packet is None and epoch == old_epoch + 1
               for roi, packet, epoch in observed.resized_states)
    assert worker.snapshot()[0][0].shape == large.shape
    assert worker.snapshot()[1] is None


def test_reconfiguration_does_not_rebind_old_selection_to_new_dimensions(monkeypatch):
    worker = ProcessingWorker(demo=True)
    small = np.full((80, 160, 3), 30, np.uint8)
    large = np.full((120, 240, 3), 30, np.uint8)
    worker.latest_frame = (small, 0.0)
    worker.configure((0, 0, 1, 1), Settings(mode="manual"))
    old_epoch = worker.epoch
    monkeypatch.setattr(fixtures, "sample_frame", lambda _: (large, None, None, []))
    packets, messages = [], []

    def message(text):
        messages.append(text)
        if "尺寸已变化" in text or "已停止" in text:
            worker.stop()

    worker.packet_ready.connect(lambda _: packets.append(worker.snapshot()[1]), Qt.ConnectionType.DirectConnection)
    worker.message.connect(message, Qt.ConnectionType.DirectConnection)
    worker.start()
    try:
        assert worker.wait(4000)
    finally:
        worker.stop()
        worker.wait(4000)
    assert any("尺寸已变化" in text for text in messages)
    assert not packets and worker.roi is None and worker.epoch == old_epoch + 1
    # A subsequent explicit selection binds normally to the latest new size.
    worker.configure((.1, .6, .8, .2), Settings(mode="manual"))
    assert worker._roi_frame_size == large.shape[:2]
