import time
import unittest

import numpy as np

from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText


class FakeOCR:
    def recognize(self, image: np.ndarray) -> list[OCRText]:
        assert image.shape == (2, 2, 3)
        return [OCRText("test", 0, 0, 10, 10, 0.9)]


class OCRWorkerTests(unittest.TestCase):
    def test_result_is_sent_to_callback(self) -> None:
        result = []
        worker = OCRWorker(
            FakeOCR(),
            on_result=lambda texts, total_ms: result.append((texts, total_ms)),
        )

        try:
            worker.submit(np.zeros((2, 2, 3), dtype=np.uint8), time.perf_counter())
            deadline = time.time() + 2
            while not result and time.time() < deadline:
                time.sleep(0.01)
        finally:
            worker.close()

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0][0].text, "test")


if __name__ == "__main__":
    unittest.main()
