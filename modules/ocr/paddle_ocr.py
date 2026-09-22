import os

import numpy as np
from paddleocr import PaddleOCR

from modules.ocr.types import OCRText


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

    def recognize(self, image: np.ndarray) -> list[OCRText]:
        results = self.ocr.predict(image)
        texts: list[OCRText] = []

        for result in results:
            result_data = result.json["res"]

            rec_texts = result_data.get("rec_texts", [])
            rec_scores = result_data.get("rec_scores", [])
            rec_boxes = result_data.get("rec_boxes", [])

            for text, score, box in zip(
                rec_texts,
                rec_scores,
                rec_boxes,
            ):
                text = text.strip()

                if not text:
                    continue

                left, top, right, bottom = map(int, box)

                texts.append(
                    OCRText(
                        text=text,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                        confidence=float(score),
                    )
                )

        return texts