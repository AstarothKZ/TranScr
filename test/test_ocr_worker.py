"""Проверки OCR-потока на простом тестовом движке (без Windows OCR)."""

import threading
import time
import unittest

import numpy as np

from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText


class FakeOCR:
    def recognize(self, image):
        return [OCRText("Hello", 0, 0, 20, 10)]


class FailingOCR:
    def recognize(self, image):
        raise RuntimeError("test OCR failure")


class OCRWorkerTests(unittest.TestCase):
    def test_success_result_is_delivered(self):
        completed = threading.Event()
        received = {}

        def on_result(texts, elapsed):
            received["texts"] = texts
            received["elapsed"] = elapsed
            completed.set()

        worker = OCRWorker(FakeOCR(), on_result=on_result)
        try:
            self.assertTrue(worker.submit(np.zeros((2, 2, 3), dtype=np.uint8), time.perf_counter()))
            self.assertTrue(completed.wait(2))
            self.assertEqual(received["texts"][0].text, "Hello")
            self.assertGreaterEqual(received["elapsed"], 0)
        finally:
            worker.close()

    def test_error_is_delivered_without_hanging(self):
        completed = threading.Event()
        received = {}

        def on_error(error):
            received["error"] = error
            completed.set()

        worker = OCRWorker(FailingOCR(), on_result=lambda *_: self.fail("unexpected result"), on_error=on_error)
        try:
            self.assertTrue(worker.submit(np.zeros((2, 2, 3), dtype=np.uint8), time.perf_counter()))
            self.assertTrue(completed.wait(2))
            self.assertIsInstance(received["error"], RuntimeError)
        finally:
            worker.close()

    def test_only_one_outstanding_job_is_accepted(self):
        entered = threading.Event()
        release = threading.Event()

        class SlowOCR:
            def recognize(self, image):
                entered.set()
                release.wait(2)
                return []

        completed = threading.Event()
        worker = OCRWorker(SlowOCR(), on_result=lambda *_: completed.set())
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.submit(image, time.perf_counter()))
            self.assertTrue(entered.wait(2))
            self.assertFalse(worker.submit(image, time.perf_counter()))
            release.set()
            self.assertTrue(completed.wait(2))
        finally:
            release.set()
            worker.close()


if __name__ == "__main__":
    unittest.main()
