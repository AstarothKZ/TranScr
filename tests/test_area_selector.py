import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from core.application import Application
from modules.capture.area_selector import AreaSelector


class AreaSelectorTests(unittest.TestCase):
    def selector(self):
        selector = AreaSelector.__new__(AreaSelector)
        selector.screen_left, selector.screen_top = -1920, -200
        selector.screen_width, selector.screen_height = 3840, 1400
        selector._start = None
        selector._rect_id = None
        selector._closed = False
        selector.canvas = Mock()
        selector.window = Mock()
        selector.on_selected = Mock()
        selector.on_cancel = Mock()
        return selector

    def test_reverse_drag_uses_absolute_desktop_coordinates(self):
        selector = self.selector()
        selector._start_drag(SimpleNamespace(x=300, y=250))
        selector._finish_drag(SimpleNamespace(x=100, y=50))
        selector.on_selected.assert_called_once_with((-1820, -150, 200, 200))
        selector.on_cancel.assert_not_called()

    def test_drag_cannot_select_pixels_outside_virtual_desktop(self):
        selector = self.selector()
        selector._start_drag(SimpleNamespace(x=-10, y=-20))
        selector._drag(SimpleNamespace(x=4000, y=1500))
        selector._finish_drag(SimpleNamespace(x=4000, y=1500))
        selector.on_selected.assert_called_once_with((-1920, -200, 3840, 1400))

    def test_small_selection_cancels_once(self):
        selector = self.selector()
        selector._start_drag(SimpleNamespace(x=100, y=100))
        selector._finish_drag(SimpleNamespace(x=105, y=105))
        selector.cancel()
        selector.on_selected.assert_not_called()
        selector.on_cancel.assert_called_once_with()

    def test_release_after_cancel_cannot_start_capture(self):
        selector = self.selector()
        selector._start_drag(SimpleNamespace(x=100, y=100))
        selector.cancel()
        selector._finish_drag(SimpleNamespace(x=300, y=300))
        selector.on_selected.assert_not_called()
        selector.on_cancel.assert_called_once_with()

    def test_saved_area_is_clipped_to_current_desktop(self):
        desktop = (-1920, -200, 3840, 1400)
        self.assertEqual(Application._restore_area((-2000, -300, 500, 400), desktop), (-1920, -200, 420, 300))
        self.assertEqual(Application._restore_area((5000, 0, 500, 400), desktop), desktop)


if __name__ == "__main__":
    unittest.main()
