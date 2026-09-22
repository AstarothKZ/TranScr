from typing import Protocol

import numpy as np

from modules.ocr.types import OCRText


class OCRModule(Protocol):
    def recognize(self, image: np.ndarray) -> list[OCRText]:
        ...