"""The facade's cache, fallback, deadlines, and lifetime are deterministic offline."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from time import perf_counter, sleep
from unittest.mock import Mock, patch

from modules.translation.batching import TranslationCancelled
from modules.translation.translator import Translator


class TranslatorTests(unittest.TestCase):
    def test_network_warmup_constructs_client_without_sending_text(self):
        provider = self.provider(["Готово"])
        translator = self.translator()
        with patch(
            "modules.translation.translator.GoogleTranslateProvider",
            return_value=provider,
        ) as factory:
            translator.warmup()
            translator.warmup()
            factory.assert_called_once()
            provider.translate_many.assert_not_called()
            self.assertEqual(translator.translate_text("Source"), "Готово")

    def translator(self, primary=None, fallback=None, **kwargs):
        translator = Translator(**kwargs)
        if primary is not None:
            translator._providers[translator.provider] = primary
        if fallback is not None:
            translator._providers[translator.fallback_provider] = fallback
        self.addCleanup(translator.close)
        return translator

    def provider(self, values):
        return Mock(translate_many=Mock(return_value=values), _client=None)

    def test_normalization_deduplication_and_cache_preserve_source_keys(self):
        primary = self.provider(["Первый текст"])
        translator = self.translator(primary)
        source = [" First\ntext ", "First text", "", "First\ntext"]
        self.assertEqual(
            translator.translate_many(source),
            {"First\ntext": "Первый текст", "First text": "Первый текст"},
        )
        self.assertEqual(primary.translate_many.call_args.args[0], ["First text"])
        self.assertEqual(translator.last_stats.unique_texts, 1)
        translator.translate_many(["First  text"])
        self.assertEqual(translator.last_stats.cache_hits, 1)
        self.assertEqual(primary.translate_many.call_count, 1)

    def test_soft_hyphen_is_removed_only_at_line_boundary(self):
        primary = self.provider(["Первый текст"])
        translator = self.translator(primary)
        translator.translate_many(["trans\u00ad\nlation"])
        self.assertEqual(primary.translate_many.call_args.args[0], ["translation"])

    def test_empty_and_invalid_input_never_starts_provider(self):
        primary = self.provider([])
        translator = self.translator(primary)
        self.assertEqual(translator.translate_many(["", " \n"]), {})
        primary.translate_many.assert_not_called()
        for value in ("Hello", b"Hello", [1]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                translator.translate_many(value)
        with self.assertRaises(TypeError):
            translator.translate_text(3)
        for kwargs in (
            {"provider": "mymemory"},
            {"provider": []},
            {"timeout": float("inf")},
            {"from_code": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Translator(**kwargs)

    def test_errors_and_invalid_values_are_not_cached(self):
        primary = self.provider([None, "Успешно", TimeoutError("slow")])
        translator = self.translator(primary)
        values = translator.translate_many(["First", "Second", "Third"])
        self.assertIsInstance(values["First"], ValueError)
        self.assertIsInstance(values["Third"], TimeoutError)
        self.assertEqual(translator.last_stats.errors, 2)
        self.assertEqual(len(translator._cache), 1)

    def test_provider_count_mismatch_never_crosses_sources(self):
        primary = self.provider(["too few"])
        translator = self.translator(primary)
        values = translator.translate_many(["First", "Second"])
        self.assertTrue(all(isinstance(value, ValueError) for value in values.values()))
        self.assertEqual(len(translator._cache), 0)

    def test_fallback_receives_only_failed_texts_and_reserves_time(self):
        primary = self.provider(["Первый", TimeoutError("slow")])
        fallback = self.provider(["Второй"])
        translator = self.translator(primary, fallback, hybrid_enabled=True, timeout=6)
        self.assertEqual(
            translator.translate_many(["First", "Second"]),
            {"First": "Первый", "Second": "Второй"},
        )
        self.assertEqual(fallback.translate_many.call_args.args[0], ["Second"])
        primary_deadline = primary.translate_many.call_args.kwargs["deadline"]
        fallback_deadline = fallback.translate_many.call_args.kwargs["deadline"]
        self.assertAlmostEqual(fallback_deadline - primary_deadline, 2.0, places=2)
        self.assertEqual(translator.last_stats.fallback_texts, 1)
        self.assertIsNotNone(translator.last_stats.expires_at)

    def test_fallback_cache_expires_and_retries_primary(self):
        primary = self.provider([RuntimeError("offline")])
        fallback = self.provider(["Перевод"])
        translator = self.translator(primary, fallback, hybrid_enabled=True)
        translator.translate_many(["Source"])
        translator.translate_many(["Source"])
        self.assertEqual(translator.last_stats.cache_hits, 1)
        self.assertEqual(translator.last_stats.fallback_texts, 1)
        self.assertEqual(translator.last_stats.incomplete, 0)
        key, entry = next(iter(translator._cache.items()))
        self.assertLessEqual(entry.expires_at - perf_counter(), 30)
        translator._cache[key] = replace(entry, expires_at=perf_counter() - 1)
        primary.translate_many.return_value = ["Основной"]
        self.assertEqual(translator.translate_text("Source"), "Основной")
        self.assertEqual(primary.translate_many.call_count, 2)
        self.assertIsNone(translator.last_stats.expires_at)

    def test_partial_cache_expiry_is_preserved_for_frame_cache(self):
        primary = self.provider(["Русский текст with untranslated words"])
        translator = self.translator(primary)
        translator.translate_many(["Source"])
        expiry = translator.last_stats.expires_at
        self.assertIsNotNone(expiry)
        self.assertEqual(translator.last_stats.incomplete, 1)
        translator.translate_many(["Source"])
        self.assertEqual(translator.last_stats.expires_at, expiry)
        self.assertEqual(translator.last_stats.incomplete, 1)
        self.assertEqual(primary.translate_many.call_count, 1)

    def test_fallback_result_cannot_be_frozen_in_frame_cache_forever(self):
        from core.pipeline import CaptureResult, FrameCache

        primary = self.provider([RuntimeError("offline")])
        fallback = self.provider(["Перевод"])
        translator = self.translator(primary, fallback, hybrid_enabled=True)
        translator.translate_many(["Source"])
        area, fingerprint = (0, 0, 10, 10), b"same frame"
        cache = FrameCache()
        result = CaptureResult(
            area, [], 1, 0, 1, 1, 1, [], expires_at=translator.last_stats.expires_at
        )
        cache.put(area, fingerprint, result)
        self.assertIsNotNone(cache.get(area, fingerprint, 0, None))
        result.expires_at = perf_counter() - 1
        self.assertIsNone(cache.get(area, fingerprint, 0, None))

    def test_cancelled_primary_never_invokes_fallback(self):
        primary = self.provider([TranslationCancelled("cancel")])
        fallback = self.provider(["wrong"])
        translator = self.translator(primary, fallback, hybrid_enabled=True)
        self.assertIsInstance(
            translator.translate_many(["Source"])["Source"], TranslationCancelled
        )
        fallback.translate_many.assert_not_called()

    def test_ready_results_survive_another_regions_deadline(self):
        primary = self.provider([])

        def work(texts, **kwargs):
            sleep(0.03)
            return ["Готово", TimeoutError("slow tail")]

        primary.translate_many.side_effect = work
        translator = self.translator(primary, timeout=0.01)
        values = translator.translate_many(["First", "Second"])
        self.assertEqual(values["First"], "Готово")
        self.assertIsInstance(values["Second"], TimeoutError)
        self.assertEqual(len(translator._cache), 1)

    def test_concurrent_same_payload_is_not_translated_twice(self):
        entered, release = Event(), Event()
        primary = self.provider([])

        def work(texts, **kwargs):
            entered.set()
            release.wait(2)
            return ["Готово"]

        primary.translate_many.side_effect = work
        translator = self.translator(primary)
        with ThreadPoolExecutor(2) as pool:
            first = pool.submit(translator.translate_many, ["Source"])
            self.assertTrue(entered.wait(1))
            second = pool.submit(translator.translate_many, ["Source"])
            release.set()
            self.assertEqual(first.result(timeout=1), second.result(timeout=1))
        self.assertEqual(primary.translate_many.call_count, 1)

    def test_waiting_caller_can_cancel_without_blocking_current_call(self):
        entered, release, cancel = Event(), Event(), Event()
        primary = self.provider([])

        def work(texts, **kwargs):
            entered.set()
            release.wait(2)
            return ["Готово"]

        primary.translate_many.side_effect = work
        translator = self.translator(primary)
        with ThreadPoolExecutor(2) as pool:
            try:
                first = pool.submit(translator.translate_many, ["First"])
                self.assertTrue(entered.wait(1))
                second = pool.submit(translator.translate_many, ["Second"], cancel)
                cancel.set()
                self.assertIsInstance(
                    second.result(timeout=0.7)["Second"], TranslationCancelled
                )
            finally:
                release.set()
            self.assertEqual(first.result(timeout=1)["First"], "Готово")
        self.assertEqual(primary.translate_many.call_count, 1)

    def test_cache_limits_and_clear(self):
        primary = self.provider([])
        primary.translate_many.side_effect = lambda texts, **kwargs: [
            "Перевод " + text for text in texts
        ]
        translator = self.translator(primary)
        translator.CACHE_SIZE = 2
        translator.translate_many(["First", "Second", "Third"])
        self.assertEqual(len(translator._cache), 2)
        self.assertEqual(
            translator._cache_chars,
            sum(
                len(key[3]) + len(entry.text)
                for key, entry in translator._cache.items()
            ),
        )
        translator.CACHE_CHARS = 1
        translator.translate_many(["Fourth"])
        self.assertEqual(translator._cache_chars, 0)
        translator.clear_cache()
        self.assertEqual(len(translator._cache), 0)

    def test_quality_pass_is_opt_in_and_one_batch(self):
        primary = self.provider(["Русский untranslated words"])
        translator = self.translator(primary)
        translator.translate_many(["Source"])
        self.assertEqual(primary.translate_many.call_count, 1)
        translator.clear_cache()
        translator.repair_incomplete = True
        primary.translate_many.side_effect = [
            ["Русский untranslated words"],
            ["переведённые слова"],
        ]
        self.assertEqual(
            translator.translate_text("Source"), "Русский переведённые слова"
        )
        self.assertEqual(primary.translate_many.call_count, 3)

    def test_close_cleans_every_provider_even_if_one_fails(self):
        primary = self.provider(["Готово"])
        fallback = self.provider([])
        translator = self.translator(primary, fallback)
        translator.translate_many(["Source"])
        primary.close.side_effect = RuntimeError("cleanup failure")
        with self.assertLogs("transcr.pipeline", level="ERROR"):
            translator.close()
        translator.close()
        primary.close.assert_called_once()
        fallback.close.assert_called_once()
        self.assertEqual(len(translator._cache), 0)
        with self.assertRaises(RuntimeError):
            translator.translate_many(["Source"])
        with self.assertRaises(RuntimeError):
            translator._get_provider("argos")


if __name__ == "__main__":
    unittest.main()
