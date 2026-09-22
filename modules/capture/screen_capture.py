from datetime import datetime
from pathlib import Path

import mss
import mss.tools
import numpy as np


class ScreenCapturer:
    """Получает изображение виртуального рабочего стола Windows."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def capture_full_screen(self) -> tuple[np.ndarray, Path]:
        """Захватывает все доступные мониторы и сохраняет PNG для тестов."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
        output_path = self.output_dir / f"screenshot_{timestamp}.png"

        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)

            image = np.frombuffer(
                screenshot.rgb,
                dtype=np.uint8,
            ).reshape(
                screenshot.height,
                screenshot.width,
                3,
            ).copy()

            mss.tools.to_png(
                screenshot.rgb,
                screenshot.size,
                output=str(output_path),
            )

        return image, output_path
