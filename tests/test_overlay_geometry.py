"""Real HWND checks; test windows remain outside the visible desktop."""
from subtitle_eraser.diagnostics import native_geometry_check


def test_real_overlay_first_show_resize_and_reshow():
    assert len(native_geometry_check()) == 4
