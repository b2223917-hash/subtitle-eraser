from threading import Condition, Event, Lock, Thread
from time import perf_counter
import numpy as np
from PySide6.QtCore import Qt

from subtitle_eraser.capture import WindowCapture
from subtitle_eraser.pipeline import Settings
from subtitle_eraser.worker import Packet, ProcessingWorker


def packet(sequence, epoch=0):
    return Packet(sequence, object(), 0.0, 0.0, (100, 50), (0, 0, 100, 50), epoch)


def test_notifications_are_bounded_and_mailbox_keeps_latest():
    worker = ProcessingWorker()
    notifications = []
    worker.packet_ready.connect(notifications.append, Qt.ConnectionType.DirectConnection)
    for sequence in range(100):
        assert worker.publish_packet(packet(sequence))
    assert notifications == [worker]
    assert worker.snapshot()[1].sequence == 99

    worker.acknowledge_packet_notification()
    assert worker.publish_packet(packet(100))
    assert notifications == [worker, worker]
    assert worker.snapshot()[1].sequence == 100


def test_reconfigure_discards_old_epoch_without_adding_queued_wakeups():
    worker = ProcessingWorker()
    notifications = []
    worker.packet_ready.connect(notifications.append, Qt.ConnectionType.DirectConnection)
    worker.publish_packet(packet(1))
    worker.configure((0, 0, 1, 1), Settings())
    assert worker.snapshot()[1] is None
    assert not worker.publish_packet(packet(2))
    assert worker.snapshot()[1] is None
    assert worker.publish_packet(packet(3, epoch=1))
    assert notifications == [worker]
    worker.acknowledge_packet_notification()
    assert worker.snapshot()[1].sequence == 3
    assert worker.publish_packet(packet(4, epoch=1))
    assert notifications == [worker, worker]


def test_stopped_worker_never_publishes_or_notifies():
    worker = ProcessingWorker()
    notifications = []
    worker.packet_ready.connect(notifications.append, Qt.ConnectionType.DirectConnection)
    worker.stop()
    assert not worker.publish_packet(packet(1))
    assert worker.snapshot()[1] is None
    assert notifications == []


def capture_mailbox():
    # Exercise the production mailbox without creating a native capture session.
    source = WindowCapture.__new__(WindowCapture)
    source.lock = Lock()
    source.frame_available = Condition(source.lock)
    source.latest = None
    source.sequence = 0
    source.closed = False
    source.error = ""
    source.control = None
    return source


def test_capture_wait_wakes_for_new_frame_and_copies_native_buffer():
    source = capture_mailbox()
    waiting = Event()
    result = []

    def wait():
        waiting.set()
        result.append(source.wait_for_frame(-1, timeout=1))

    thread = Thread(target=wait)
    thread.start()
    assert waiting.wait(1)
    frame = type("Frame", (), {"frame_buffer": np.full((8, 12, 4), 23, np.uint8)})()
    source._on_frame(frame, None)
    thread.join(1)
    assert not thread.is_alive()
    assert result[0][0] == 1
    frame.frame_buffer[:] = 0
    assert np.all(result[0][1] == 23)
    assert source.wait_for_frame(1, timeout=0) is result[0]


def test_capture_close_wakes_waiter_without_waiting_for_timeout():
    source = capture_mailbox()
    waiting = Event()
    result = []

    def wait():
        waiting.set()
        result.append(source.wait_for_frame(-1, timeout=10))

    thread = Thread(target=wait)
    thread.start()
    assert waiting.wait(1)
    source.stop()
    thread.join(.5)
    assert not thread.is_alive()
    assert result == [None]


def test_ready_signal_updates_ui_without_waiting_for_poll_timer(monkeypatch):
    from PySide6.QtCore import QThread
    from PySide6.QtWidgets import QApplication
    from subtitle_eraser import ui

    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(ui.windows, "list_windows", lambda: [])
    monkeypatch.setattr(ui.windows, "register_toggle", lambda _: True)
    monkeypatch.setattr(ui.windows, "unregister_toggle", lambda _: None)
    monkeypatch.setattr(ui.windows, "exclude_from_capture", lambda _: True)
    window = ui.MainWindow()
    window.timer.stop()
    threads = []
    original_poll = window.poll

    def observe():
        threads.append(QThread.currentThread())
        original_poll()

    window.poll = observe
    try:
        # Exercise the actual worker -> queued Qt slot -> poll path, using only
        # generated frames. Neither the main window nor the overlay is shown.
        window.start_demo()
        deadline = perf_counter() + 4
        while window.packet is None and perf_counter() < deadline:
            app.processEvents()
            Event().wait(.005)
        assert window.packet is not None
        assert threads and all(thread == app.thread() for thread in threads)
        assert window.packet.result.timing["ocr_ms"] == 0
    finally:
        window.close()
