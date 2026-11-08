"""Efficient capture of the selected part of the Windows virtual desktop."""

import threading
from datetime import datetime
from pathlib import Path

import mss
import mss.tools
import numpy as np

from .types import ScreenBounds


class ScreenCapturer:
    """Captures one rectangle; MSS is reused between calls where possible."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.virtual_bounds = self.get_virtual_screen_bounds()
        self.last_bounds: ScreenBounds = self.virtual_bounds
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
        *,
        save_debug: bool = False,
    ) -> tuple[np.ndarray, Path | None]:
        """Capture an absolute desktop rectangle and return RGB pixels plus an optional PNG path."""
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
            sct = self._sct

        screenshot = sct.grab(monitor)
        actual_bounds = (
            left,
            top,
            int(screenshot.width),
            int(screenshot.height),
        )
        self.last_bounds = actual_bounds
        # The NumPy view keeps the screenshot's byte buffer alive and avoids a
        # second full-frame copy. OCR engines in this project only read pixels.
        rgb = screenshot.rgb
        image = np.frombuffer(rgb, dtype=np.uint8).reshape(
            screenshot.height, screenshot.width, 3
        )

        output_path = None
        if save_debug:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
            output_path = self.output_dir / f"screenshot_{timestamp}.png"
            mss.tools.to_png(rgb, screenshot.size, output=str(output_path))

        return image, output_path

    def close(self) -> None:
        with self._sct_lock:
            if self._sct is not None:
                try:
                    self._sct.close()
                finally:
                    self._sct = None
