import unittest
from queue import Queue
from time import perf_counter
from unittest.mock import Mock

import numpy as np

from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText


class OCRWorkerTests(unittest.TestCase):
    def test_worker_recovers_after_recognition_error(self):
        texts = [OCRText("Hello", 0, 0, 20, 10)]
        error = RuntimeError("OCR failed")
        engine = Mock(recognize=Mock(side_effect=[texts, error, texts]))
        events = Queue()
        worker = OCRWorker(
            engine,
            on_result=lambda result, elapsed, token: events.put((result, elapsed, token)),
            on_error=lambda failure, token: events.put((failure, 0, token)),
        )
        self.addCleanup(worker.close)
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        for expected in (texts, error, texts):
            token = perf_counter()
            self.assertTrue(worker.submit(image, token))
            result, elapsed, returned_token = events.get(timeout=2)
            self.assertIs(result, expected)
            self.assertEqual(returned_token, token)
            self.assertGreaterEqual(elapsed, 0)
        self.assertEqual(engine.recognize.call_count, 3)

    def test_callback_error_does_not_leave_worker_busy(self):
        events = Queue()
        calls = 0

        def on_result(texts, elapsed, token):
            nonlocal calls
            calls += 1
            events.put(token)
            if calls == 1:
                raise RuntimeError("Callback failed")

        worker = OCRWorker(Mock(recognize=Mock(return_value=[])), on_result)
        self.addCleanup(worker.close)
        image = np.zeros((10, 20, 3), dtype=np.uint8)
        with self.assertLogs("modules.ocr.ocr_worker", level="ERROR") as logs:
            for _ in range(2):
                token = perf_counter()
                self.assertTrue(worker.submit(image, token))
                self.assertEqual(events.get(timeout=2), token)
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
