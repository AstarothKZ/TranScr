import os

import numpy as np
from paddleocr import PaddleOCR


class PaddleOCRModule:
    """Распознаёт текст на изображении с помощью PP-OCRv6."""

    def __init__(self) -> None:
        cpu_count = os.cpu_count() or 4
        cpu_threads = min(8, cpu_count)

        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            device="cpu",
            enable_mkldnn=True,
            cpu_threads=cpu_threads,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: np.ndarray) -> list[str]:
        """Возвращает распознанные непустые строки."""
        results = self.ocr.predict(image)
        texts: list[str] = []

        for result in results:
            result_data = result.json["res"]

            for text in result_data["rec_texts"]:
                if text.strip():
                    texts.append(text.strip())

        return texts
