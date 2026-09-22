import os

import numpy as np
from paddleocr import PaddleOCR

from modules.ocr.types import OCRText


class PaddleOCRModule:
    """Альтернативный OCR через PaddleOCR 3.x."""

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
            data = result.json["res"]
            rec_texts = data.get("rec_texts", [])
            rec_scores = data.get("rec_scores", [])
            rec_boxes = data.get("rec_boxes", [])
            rec_polys = data.get("rec_polys", [])

            for index, raw_text in enumerate(rec_texts):
                text = str(raw_text).strip()
                if not text:
                    continue

                polygon = None
                left = top = right = bottom = 0

                if index < len(rec_polys):
                    points = np.asarray(rec_polys[index], dtype=float)
                    polygon = tuple(
                        (int(point[0]), int(point[1])) for point in points
                    )
                    left = min(point[0] for point in polygon)
                    top = min(point[1] for point in polygon)
                    right = max(point[0] for point in polygon)
                    bottom = max(point[1] for point in polygon)
                elif index < len(rec_boxes):
                    box = rec_boxes[index]
                    left, top, right, bottom = map(int, box)

                confidence = (
                    float(rec_scores[index])
                    if index < len(rec_scores)
                    else None
                )

                texts.append(
                    OCRText(
                        text=text,
                        left=left,
                        top=top,
                        right=right,
                        bottom=bottom,
                        confidence=confidence,
                        polygon=polygon,
                    )
                )

        return texts
