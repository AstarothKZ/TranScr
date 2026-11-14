"""Efficient capture of the selected part of the Windows virtual desktop."""

import threading

import mss
import numpy as np

from .types import ScreenBounds


class ScreenCapturer:
    """Captures one rectangle; MSS is reused between calls where possible."""

    def __init__(self) -> None:
        self.virtual_bounds = self.get_virtual_screen_bounds()
        self._sct = None
        self._sct_lock = threading.Lock()

    @staticmethod
    def get_virtual_screen_bounds() -> ScreenBounds:
        """Return virtual-screen bounds, including negative multi-monitor offsets."""
        with mss.MSS() as sct:
            monitor = sct.monitors[0]
            return (
                int(monitor["left"]),
                int(monitor["top"]),
                int(monitor["width"]),
                int(monitor["height"]),
            )

    def capture_region(
        self,
        bounds: ScreenBounds | None = None,
    ) -> np.ndarray:
        """Capture an absolute desktop rectangle and return RGB pixels."""
        left, top, width, height = bounds or self.virtual_bounds
        left, top, width, height = int(left), int(top), int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError(
                f"Некорректная область захвата: {(left, top, width, height)}"
            )

        monitor = {"left": left, "top": top, "width": width, "height": height}
        with self._sct_lock:
            if self._sct is None:
                # The translation pipeline is serialized, so this MSS instance
                # is reused by the same worker for subsequent captures.
                self._sct = mss.MSS()
            # Keep close() from releasing the backend between lookup and grab.
            screenshot = self._sct.grab(monitor)

        # MSS supplies BGRA. Copy directly into one contiguous RGB allocation,
        # avoiding the sliced byte buffers and extra bytes copy in .rgb.
        bgra = np.asarray(screenshot)
        image = np.empty((screenshot.height, screenshot.width, 3), dtype=np.uint8)
        image[..., 0] = bgra[..., 2]
        image[..., 1] = bgra[..., 1]
        image[..., 2] = bgra[..., 0]
        image.setflags(write=False)
        return image

    def close(self) -> None:
        with self._sct_lock:
            if self._sct is not None:
                try:
                    self._sct.close()
                finally:
                    self._sct = None
