import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import numpy as np
from mss.screenshot import ScreenShot

from modules.capture.screen_capture import ScreenCapturer


class ScreenCaptureTests(unittest.TestCase):
    def capturer(self, backend):
        with patch.object(
            ScreenCapturer, "get_virtual_screen_bounds", return_value=(-1920, -200, 3840, 1400)
        ):
            capturer = ScreenCapturer()
        factory = self.enterContext(
            patch("modules.capture.screen_capture.mss.MSS", return_value=backend)
        )
        self.addCleanup(capturer.close)
        return capturer, factory

    def test_rgb_pixels_are_contiguous_and_independent_of_source_buffer(self):
        raw = bytearray((3, 2, 1, 0, 6, 5, 4, 255, 9, 8, 7, 0, 12, 11, 10, 255))
        screenshot = ScreenShot.from_size(raw, 2, 2)
        backend = Mock(grab=Mock(return_value=screenshot))
        capturer, _ = self.capturer(backend)
        image = capturer.capture_region((-10, -5, 2, 2))
        expected = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8)
        raw[:] = bytes(len(raw))
        np.testing.assert_array_equal(image, expected)
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(image.flags.c_contiguous)
        backend.grab.assert_called_once_with({"left": -10, "top": -5, "width": 2, "height": 2})

    def test_backend_is_reused_and_can_reopen_after_close(self):
        backend = Mock(grab=Mock(return_value=ScreenShot.from_size(bytearray(4), 1, 1)))
        capturer, factory = self.capturer(backend)
        factory.assert_not_called()
        capturer.capture_region((0, 0, 1, 1))
        capturer.capture_region((0, 0, 1, 1))
        factory.assert_called_once_with()
        capturer.close()
        capturer.close()
        backend.close.assert_called_once_with()
        capturer.capture_region((0, 0, 1, 1))
        self.assertEqual(factory.call_count, 2)

    def test_invalid_dimensions_do_not_open_backend(self):
        capturer, factory = self.capturer(Mock())
        for bounds in ((0, 0, 0, 20), (0, 0, 20, 0), (0, 0, -1, 20), (0, 0, 20, -1)):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                capturer.capture_region(bounds)
        factory.assert_not_called()

    def test_virtual_desktop_includes_negative_monitor_offsets(self):
        backend = Mock()
        backend.__enter__ = Mock(return_value=backend)
        backend.__exit__ = Mock(return_value=False)
        backend.monitors = [{"left": -1920, "top": -200, "width": 3840, "height": 1400}]
        with patch("modules.capture.screen_capture.mss.MSS", return_value=backend):
            self.assertEqual(ScreenCapturer.get_virtual_screen_bounds(), (-1920, -200, 3840, 1400))
        backend.__exit__.assert_called_once()

    def test_close_waits_for_capture_to_finish(self):
        entered, release, closing, closed = (threading.Event() for _ in range(4))

        def grab(_monitor):
            entered.set()
            if not release.wait(2):
                raise TimeoutError("Test capture was not released")
            if closed.is_set():
                raise RuntimeError("Backend closed during capture")
            return ScreenShot.from_size(bytearray(4), 1, 1)

        backend = Mock(grab=grab, close=closed.set)
        capturer, _ = self.capturer(backend)

        def close():
            closing.set()
            capturer.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            capture = executor.submit(capturer.capture_region, (0, 0, 1, 1))
            try:
                self.assertTrue(entered.wait(2))
                closure = executor.submit(close)
                self.assertTrue(closing.wait(2))
                self.assertFalse(closed.wait(0.05))
            finally:
                release.set()
            self.assertEqual(capture.result(timeout=2).shape, (1, 1, 3))
            closure.result(timeout=2)
        self.assertTrue(closed.is_set())


if __name__ == "__main__":
    unittest.main()
