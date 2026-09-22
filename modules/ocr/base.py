from typing import Protocol

import numpy as np

from modules.ocr.types import OCRText


class OCREngine(Protocol):
    """Минимальный интерфейс любого OCR-модуля."""

    def recognize(self, image: np.ndarray) -> list[OCRText]:
        ...
