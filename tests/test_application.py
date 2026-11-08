import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

from core.application import Application


class ApplicationTests(unittest.TestCase):
    @patch("core.application.AreaSelector")
    def test_quick_selection_starts_capture_without_saving(self, selector):
        app = Application.__new__(Application)
        app._active = True
        app._busy = app._starting = False
        app._area_selector = None
        app._display_generation = 0
        app.selected_area = (0, 0, 1920, 1080)
        app.full_screen_bounds = app.selected_area
        app.overlay = Mock()
        app.root = Mock()
        app.status_var = Mock()
        app.config = Mock()
        app._start_capture = Mock()
        app.quick_select_area()
        bounds = (100, 200, 300, 140)
        selector.call_args.kwargs["on_selected"](bounds)
        app._start_capture.assert_called_once_with("quick-area", area_override=bounds)
        self.assertEqual(app.selected_area, (0, 0, 1920, 1080))
        app.config.save.assert_not_called()

    @patch("core.application.AreaSelector")
    def test_quick_selection_cancel_does_not_capture(self, selector):
        app = Application.__new__(Application)
        app._active = True
        app._busy = app._starting = False
        app._area_selector = None
        app._display_generation = 0
        app.full_screen_bounds = (0, 0, 1920, 1080)
        app.overlay = app.root = app.status_var = Mock()
        app._start_capture = Mock()
        app.quick_select_area()
        selector.call_args.kwargs["on_cancel"]()
        app._start_capture.assert_not_called()
        self.assertIsNone(app._area_selector)

    def test_keyboard_thread_does_not_access_tk(self):
        app = Application.__new__(Application)
        app._active = True
        app._capture_hotkey_text = "alt+t"
        app._events = Queue()
        app._area_selector = None
        app.dismiss_mode_var = Mock()
        app.dismiss_mode_var.get.side_effect = AssertionError("Tk accessed from hook")
        app._dismiss_unbound_key(SimpleNamespace(event_type="down", name="x"))
        self.assertEqual(app._events.get_nowait(), ("dismiss", "x", True))

    def test_runtime_change_invalidates_frame_cache(self):
        app = Application.__new__(Application)
        app._frame_cache = Mock()
        app.start_button = Mock()
        app.select_area_button = Mock()
        app._activate_hotkeys = Mock()
        runtime = Mock(argos_model_ready=True)
        app._handle_runtime_ready(runtime)
        app._frame_cache.clear.assert_called_once()
        self.assertIs(app._runtime, runtime)

    def test_stop_before_capture_timer_prevents_submission(self):
        app = Application.__new__(Application)
        app._shutting_down = False
        app._active = False
        app._busy = True
        app._capture_executor = Mock()
        app._submit_capture(1, 0, (0, 0, 100, 100), False, Mock())
        self.assertFalse(app._busy)
        app._capture_executor.submit.assert_not_called()
