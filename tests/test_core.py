import json
import tempfile
import threading
import unittest
from pathlib import Path
from queue import Queue
from unittest.mock import Mock

import numpy as np

from core.config import AppConfig
from core.pipeline import FrameCache, RuntimeBundle, process_capture
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText
from modules.text_processing.filters import should_translate
from modules.text_processing.text_regions import group_ocr_texts


class CoreTests(unittest.TestCase):
    def test_malformed_config_values_use_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(Path(directory))
            config.path = Path(directory) / "settings.json"
            config.path.write_text(
                json.dumps(
                    {
                        "provider": [],
                        "ocr": {},
                        "fallback_provider": [],
                        "dismiss_mode": {},
                        "area": [True, 0, 100, 100],
                    }
                )
            )
            self.assertEqual(config.load(), AppConfig.DEFAULTS)

    def test_numeric_filter_keeps_real_sentences(self):
        for value in ["123", "2026-11-05", "A-4SU"]:
            self.assertFalse(should_translate(value))
        self.assertTrue(should_translate("The A-4SU entered the room"))

    def test_grouping_keeps_columns_separate(self):
        lines = [
            OCRText("Left first", 0, 0, 100, 20),
            OCRText("Right first", 200, 0, 300, 20),
            OCRText("Left next", 0, 24, 100, 44),
            OCRText("Right next", 200, 24, 300, 44),
        ]
        groups = group_ocr_texts(lines)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].text, "Left first\nLeft next")

    def runtime(self):
        results = Queue()
        ocr = Mock()

        def submit(image, started):
            results.put(("result", [OCRText("Hello", 0, 0, 80, 20)], 1.0, started))
            return True

        ocr.submit.side_effect = submit
        translator = Mock()
        translator.translate_many.return_value = {"Hello": "Привет"}
        return RuntimeBundle(
            ("windows", "mymemory", False, "argos"), ocr, results, translator
        )

    def test_exact_frame_cache_and_invalidation(self):
        capturer = Mock()
        capturer.capture_region.return_value = (
            np.indices((20, 80, 3))[1].astype(np.uint8),
            None,
        )
        runtime = self.runtime()
        cache = FrameCache()
        area = (0, 0, 80, 20)
        first = process_capture(capturer, area, False, runtime, cache)
        second = process_capture(capturer, area, False, runtime, cache)
        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(runtime.ocr_worker.submit.call_count, 1)
        cache.clear()
        process_capture(capturer, area, False, runtime, cache)
        self.assertEqual(runtime.ocr_worker.submit.call_count, 2)

    def test_failed_translation_is_not_frame_cached(self):
        capturer = Mock()
        capturer.capture_region.return_value = (
            np.indices((20, 80, 3))[1].astype(np.uint8),
            None,
        )
        runtime = self.runtime()
        runtime.translator.translate_many.return_value = {
            "Hello": RuntimeError("network")
        }
        cache = FrameCache()
        for _ in range(2):
            process_capture(capturer, (0, 0, 80, 20), False, runtime, cache)
        self.assertEqual(runtime.ocr_worker.submit.call_count, 2)

    def test_cancel_after_ocr_skips_translation(self):
        capturer = Mock()
        capturer.capture_region.return_value = (
            np.indices((20, 80, 3))[1].astype(np.uint8),
            None,
        )
        runtime = self.runtime()
        cancel = threading.Event()
        cancel.set()
        process_capture(capturer, (0, 0, 80, 20), False, runtime, cancel=cancel)
        runtime.translator.translate_many.assert_not_called()

    def test_cancel_before_capture_does_no_work(self):
        cancel = threading.Event()
        cancel.set()
        capturer = Mock()
        runtime = self.runtime()
        process_capture(capturer, (0, 0, 80, 20), False, runtime, cancel=cancel)
        capturer.capture_region.assert_not_called()
        runtime.ocr_worker.submit.assert_not_called()

    def test_cancel_during_ocr_wait_returns_without_callback(self):
        cancel = threading.Event()
        capturer = Mock()
        capturer.capture_region.return_value = (np.zeros((20, 80, 3), dtype=np.uint8), None)
        runtime = self.runtime()
        runtime.ocr_worker.submit.side_effect = lambda *_: (cancel.set() or True)
        result = process_capture(capturer, (0, 0, 80, 20), False, runtime, cancel=cancel)
        self.assertEqual(result.translated_regions, [])
        runtime.translator.translate_many.assert_not_called()

    def test_cancel_during_translation_does_not_cache_partial_frame(self):
        cancel = threading.Event()
        capturer = Mock()
        capturer.capture_region.return_value = (np.indices((20, 80, 3))[1].astype(np.uint8), None)
        runtime = self.runtime()
        runtime.translator.translate_many.side_effect = lambda *_: (cancel.set() or {"Hello": "Привет"})
        cache = FrameCache()
        process_capture(capturer, (0, 0, 80, 20), False, runtime, cache, cancel)
        cancel.clear()
        runtime.translator.translate_many.side_effect = None
        result = process_capture(capturer, (0, 0, 80, 20), False, runtime, cache, cancel)
        self.assertFalse(result.cache_hit)
        self.assertEqual(runtime.ocr_worker.submit.call_count, 2)

    def test_partial_language_result_does_not_enter_frame_cache(self):
        capturer = Mock()
        capturer.capture_region.return_value = (np.indices((20, 80, 3))[1].astype(np.uint8), None)
        runtime = self.runtime()
        runtime.translator.translate_many.return_value = {"Hello": "Привет very limited"}
        cache = FrameCache()
        for _ in range(2):
            result = process_capture(capturer, (0, 0, 80, 20), False, runtime, cache)
            self.assertFalse(result.cache_hit)

    def test_frame_fingerprint_accepts_noncontiguous_image(self):
        frame = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)[:, ::2]
        self.assertEqual(FrameCache.fingerprint(frame), FrameCache.fingerprint(frame.copy()))

    def test_background_prepared_and_reused_with_frame(self):
        capturer = Mock()
        capturer.capture_region.return_value = (np.indices((20, 80, 3))[1].astype(np.uint8), None)
        runtime = self.runtime()
        cache = FrameCache()
        first = process_capture(capturer, (0, 0, 80, 20), False, runtime, cache)
        second = process_capture(capturer, (0, 0, 80, 20), False, runtime, cache)
        self.assertIn((0, 0, 80, 20), first.backgrounds)
        self.assertIs(first.backgrounds, second.backgrounds)
        self.assertEqual(second.background_ms, 0)

    def test_worker_rejects_overlapping_frames(self):
        started, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        engine = Mock()

        def recognize(image):
            started.set()
            release.wait(2)
            return []

        engine.recognize.side_effect = recognize
        worker = OCRWorker(engine, lambda texts, elapsed, token: finished.set())
        try:
            self.assertTrue(worker.submit(np.zeros((1, 1, 3)), 0))
            self.assertTrue(started.wait(2))
            self.assertFalse(worker.submit(np.zeros((1, 1, 3)), 0))
            release.set()
            self.assertTrue(finished.wait(2))
        finally:
            release.set()
            worker.close()

    def test_late_result_from_previous_ocr_is_ignored(self):
        capturer = Mock()
        capturer.capture_region.return_value = (
            np.indices((20, 80, 3))[1].astype(np.uint8),
            None,
        )
        runtime = self.runtime()

        def submit(image, started):
            runtime.ocr_results.put(("error", RuntimeError("stale"), 0, started - 1))
            runtime.ocr_results.put(
                ("result", [OCRText("Hello", 0, 0, 80, 20)], 1, started)
            )
            return True

        runtime.ocr_worker.submit.side_effect = submit
        result = process_capture(capturer, (0, 0, 80, 20), False, runtime)
        self.assertEqual(result.translated_regions[0][1], "Привет")


if __name__ == "__main__":
    unittest.main()
