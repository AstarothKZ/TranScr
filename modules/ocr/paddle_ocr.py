import os
import math

import numpy as np
from paddleocr import PaddleOCR

from modules.ocr.types import OCRText


class PaddleOCRModule:
    """Альтернативный OCR через PaddleOCR 3.x."""

    def __init__(self) -> None:
        cpu_count = os.cpu_count() or 4
        self.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv6_small_det",
            text_recognition_model_name="PP-OCRv6_small_rec",
            # Default detector expansion includes neighbouring image/text in
            # large headings. A tighter crop keeps overlay geometry local.
            text_det_unclip_ratio=1.0,
            device="cpu",
            cpu_threads=min(8, cpu_count),
            enable_mkldnn=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def recognize(self, image: np.ndarray, *, _split_labels=True) -> list[OCRText]:
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
                    polygon = tuple((int(point[0]), int(point[1])) for point in points)
                    if polygon:
                        left = min(point[0] for point in polygon)
                        top = min(point[1] for point in polygon)
                        right = math.ceil(max(point[0] for point in points))
                        bottom = math.ceil(max(point[1] for point in points))
                elif index < len(rec_boxes):
                    left, top, right, bottom = map(int, rec_boxes[index])

                # Detection is kept tight for recognition. Add only a small
                # display safety margin for antialiased glyph edges.
                left, top = max(0, left-2), max(0, top-2)
                right = min(image.shape[1], right+2)
                bottom = min(image.shape[0], bottom+3)
                confidence = (
                    float(rec_scores[index]) if index < len(rec_scores) else None
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
        if _split_labels:
            from modules.ocr.labels import recover_labels
            return recover_labels(image, texts, lambda crop: self.recognize(crop, _split_labels=False))
        return texts
