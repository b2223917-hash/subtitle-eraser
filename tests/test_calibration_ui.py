"""Calibration/profile controls with hidden Qt widgets and synthetic frames only."""
from time import perf_counter
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QDialog

from subtitle_eraser import ui
from subtitle_eraser.profiles import ProfileStore
from subtitle_eraser.style import SubtitleStyle
from subtitle_eraser.worker import ProcessingWorker


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def configured(qt_app, monkeypatch, tmp_path):
    # All widgets remain hidden; these mocks also prevent native hotkeys or
    # window enumeration from touching the user's current desktop session.
    monkeypatch.setattr(ui.windows, "list_windows", lambda: [])
    monkeypatch.setattr(ui.windows, "register_toggle", lambda _: True)
    monkeypatch.setattr(ui.windows, "unregister_toggle", lambda _: None)
    monkeypatch.setattr(ui.windows, "exclude_from_capture", lambda _: True)
    store = ProfileStore(tmp_path / "profiles.json")
    window = ui.MainWindow(profile_store=store)
    window.timer.stop()
    worker = ProcessingWorker(demo=True)
    frame = np.full((240, 640, 3), 25, np.uint8)
    worker.latest_frame = frame, perf_counter()
    worker.configure((.10, .65, .80, .20), window.settings())
    window.worker = worker
    window.player_identity = r"C:\SyntheticPlayer\player.exe"
    state = SimpleNamespace(window=window, worker=worker, store=store, frame=frame)
    try:
        yield state
    finally:
        window.close()
        window.overlay.deleteLater()
        window.deleteLater()
        qt_app.sendPostedEvents()


def set_calibrated_style(state):
    style = SubtitleStyle(25, 220, 245, glyph_height=26)
    state.window.subtitle_style = style
    state.window.update_style_label()
    state.window.reconfigure()
    return style


def save_settings(state):
    window = state.window
    window.color_combo.setCurrentIndex(window.color_combo.findData("yellow"))
    window.padding.setValue(4)
    window.continuity_check.setChecked(False)
    style = set_calibrated_style(state)
    window.save_profile()
    assert state.store.path.exists()
    return style


def current_state(window):
    return (window.settings(), window.worker.roi, window.worker.epoch,
            window.continuity_check.isChecked(), window.packet)


def test_save_and_restore_uses_confirmed_roi_and_one_atomic_configuration(configured, monkeypatch):
    state = configured
    window, worker = state.window, state.worker
    assert not state.store.path.exists()  # Merely opening the app does not write.
    saved_roi = worker.roi
    style = save_settings(state)
    saved = ProfileStore(state.store.path).lookup(window.player_identity, (640, 240))
    assert saved["settings"]["style"] == style.to_dict()
    assert saved["settings"]["edge_padding"] == 4
    assert saved["settings"]["continuity"] is False

    window.mode_combo.setCurrentIndex(window.mode_combo.findData("auto"))
    window.color_combo.setCurrentIndex(window.color_combo.findData("white"))
    window.padding.setValue(1)
    window.continuity_check.setChecked(True)
    worker.configure((.2, .5, .6, .3), window.settings())
    before = current_state(window)
    adjusted_roi = (.12, .67, .75, .18)
    visited = []

    def accept(dialog):
        visited.append(True)
        assert window.paused
        assert current_state(window) == before
        assert dialog.canvas.normalized_region() == pytest.approx(saved_roi)
        dialog.canvas.set_normalized_region(adjusted_roi)
        return QDialog.Accepted

    monkeypatch.setattr(ui.SelectionDialog, "exec", accept)
    window.restore_profile()
    assert visited == [True]
    assert not window.paused
    assert worker.roi == adjusted_roi
    assert worker.epoch == before[2] + 1
    assert window.settings().mode == "manual"
    assert window.settings().font_color == "yellow"
    assert window.settings().edge_padding == 4
    assert window.settings().style == style
    assert worker.settings == window.settings()
    assert not window.continuity_check.isChecked()
    assert window.continuity_check.isEnabled()


@pytest.mark.parametrize("paused", [False, True])
def test_cancel_restore_preserves_current_settings_and_pause_state(configured, monkeypatch, paused):
    state = configured
    save_settings(state)
    window = state.window
    window.padding.setValue(1)
    window.color_combo.setCurrentIndex(window.color_combo.findData("white"))
    window.paused = paused
    before = current_state(window)
    monkeypatch.setattr(ui.SelectionDialog, "exec", lambda _: QDialog.Rejected)
    window.restore_profile()
    assert current_state(window) == before
    assert window.paused is paused


def test_restore_rejects_different_frame_size_before_opening_dialog(configured, monkeypatch):
    state = configured
    save_settings(state)
    state.worker.latest_frame = np.zeros((240, 641, 3), np.uint8), perf_counter()
    before = current_state(state.window)

    def unexpected_dialog(_):
        pytest.fail("A stale saved rectangle must not open a restore preview")

    monkeypatch.setattr(ui.SelectionDialog, "exec", unexpected_dialog)
    state.window.restore_profile()
    assert current_state(state.window) == before
    assert "尺寸" in state.window.preference_note.text()


def test_restore_rejects_resize_while_confirmation_is_open(configured, monkeypatch):
    state = configured
    save_settings(state)
    state.window.padding.setValue(1)
    before = current_state(state.window)

    def accept_after_resize(_):
        state.worker.latest_frame = np.zeros((250, 640, 3), np.uint8), perf_counter()
        return QDialog.Accepted

    monkeypatch.setattr(ui.SelectionDialog, "exec", accept_after_resize)
    state.window.restore_profile()
    assert current_state(state.window) == before
    assert not state.window.paused
    assert "尚未应用" in state.window.preference_note.text()


def test_color_preset_clears_calibration_but_padding_preserves_it(configured):
    state = configured
    style = set_calibrated_style(state)
    roi = state.worker.roi
    state.window.padding.setValue(5)
    assert state.window.subtitle_style == style
    assert state.worker.settings.style == style
    assert state.worker.settings.edge_padding == 5
    assert state.worker.roi == roi
    state.window.color_combo.setCurrentIndex(state.window.color_combo.findData("white"))
    assert state.window.subtitle_style is None
    assert state.worker.settings.style is None
    assert not state.window.reset_style_button.isEnabled()
    assert state.worker.roi == roi


def test_changing_source_clears_sampled_style_without_starting_capture(configured, monkeypatch):
    state = configured
    set_calibrated_style(state)
    old_worker = state.worker
    monkeypatch.setattr(ProcessingWorker, "start", lambda _: None)
    state.window.start_source(demo=True)
    assert state.window.worker is not old_worker
    assert not state.window.worker.isRunning()
    assert state.window.subtitle_style is None
    assert state.window.settings().style is None
    assert state.window.player_identity is None
    assert old_worker.stop_event.is_set()


def test_selection_preview_preserves_normalized_rectangle_when_resized(qt_app):
    image = np.zeros((360, 640, 3), np.uint8)
    roi = (.11, .67, .77, .16)
    dialog = ui.SelectionDialog(image, None, manual=True, initial_roi=roi)
    try:
        assert dialog.ok.isEnabled()
        for width, height in ((700, 340), (1000, 400), (700, 700)):
            dialog.canvas.resize(width, height)
            bounds = dialog.canvas.image_rect()
            selected = dialog.canvas.selected
            assert dialog.canvas.normalized_region() == pytest.approx(roi)
            assert (selected.x() - bounds.x()) / bounds.width() == pytest.approx(roi[0])
            assert (selected.y() - bounds.y()) / bounds.height() == pytest.approx(roi[1])
            assert selected.width() / bounds.width() == pytest.approx(roi[2])
            assert selected.height() / bounds.height() == pytest.approx(roi[3])
    finally:
        dialog.close()


def test_calibration_dialog_requires_valid_pick_and_invalid_pick_clears_previous(qt_app):
    image = np.full((80, 180, 3), 25, np.uint8)
    cv2.line(image, (40, 20), (40, 44), (245, 245, 245), 2)
    cv2.line(image, (30, 32), (50, 32), (245, 245, 245), 2)
    original = image.copy()
    dialog = ui.StyleCalibrationDialog(image)
    try:
        assert not dialog.ok.isEnabled()
        dialog.pick(40, 25)
        assert dialog.style is not None
        assert dialog.ok.isEnabled()
        assert dialog.mask_preview.image is not None
        assert dialog.repair_preview.image is not None
        dialog.pick(4, 4)
        assert dialog.style is None
        assert not dialog.ok.isEnabled()
        assert "明亮笔画" in dialog.note.text()
        dialog.pick(40, 25)
        assert dialog.ok.isEnabled()
        assert np.array_equal(image, original)
    finally:
        dialog.close()


def test_calibration_applies_only_after_acceptance_and_does_not_save_automatically(configured, monkeypatch):
    state = configured
    style = SubtitleStyle(0, 0, 245)
    before_epoch = state.worker.epoch
    roi = state.worker.roi

    def accept(dialog):
        assert state.window.paused
        assert state.window.subtitle_style is None
        dialog.style = style
        return QDialog.Accepted

    monkeypatch.setattr(ui.StyleCalibrationDialog, "exec", accept)
    state.window.calibrate_subtitle_style()
    assert state.window.subtitle_style == style
    assert state.worker.settings.style == style
    assert state.worker.roi == roi
    assert state.worker.epoch == before_epoch + 1
    assert not state.window.paused
    assert not state.store.path.exists()


def test_calibration_rejects_selection_changed_during_dialog(configured, monkeypatch):
    state = configured

    def accept_after_roi_changes(dialog):
        dialog.style = SubtitleStyle(0, 0, 245)
        state.worker.configure((.2, .6, .6, .2), state.window.settings())
        return QDialog.Accepted

    monkeypatch.setattr(ui.StyleCalibrationDialog, "exec", accept_after_roi_changes)
    state.window.calibrate_subtitle_style()
    assert state.window.subtitle_style is None
    assert state.worker.settings.style is None
    assert not state.window.paused
    assert "已变化" in state.window.preference_note.text()


def test_region_selection_survives_resize_while_dialog_is_open(configured, monkeypatch):
    state = configured
    set_calibrated_style(state)
    before = current_state(state.window)

    def accept_after_resize(dialog):
        assert state.window.paused
        dialog.canvas.set_normalized_region((.2, .6, .6, .2))
        state.worker.latest_frame = np.zeros((250, 640, 3), np.uint8), perf_counter()
        return QDialog.Accepted

    monkeypatch.setattr(ui.SelectionDialog, "exec", accept_after_resize)
    state.window.select_region()
    assert state.worker.roi is None  # Never apply a fullscreen selection to the changed size.
    assert state.worker.settings.style is None
    assert state.worker.epoch > before[2]
    assert not state.window.paused
    assert "已保存" in state.window.preference_note.text()
    restored, _, _ = state.worker._accept_frame(state.frame, perf_counter())
    assert restored == (.2, .6, .6, .2)


def test_accepting_new_region_clears_previous_sampled_style(configured, monkeypatch):
    state = configured
    set_calibrated_style(state)
    before_epoch = state.worker.epoch
    new_roi = (.2, .6, .6, .2)

    def accept(dialog):
        assert state.window.paused
        dialog.canvas.set_normalized_region(new_roi)
        return QDialog.Accepted

    monkeypatch.setattr(ui.SelectionDialog, "exec", accept)
    state.window.select_region()
    assert state.worker.roi == new_roi
    assert state.worker.epoch == before_epoch + 1
    assert state.window.subtitle_style is None
    assert state.worker.settings.style is None
    assert not state.window.reset_style_button.isEnabled()
    assert not state.window.paused
