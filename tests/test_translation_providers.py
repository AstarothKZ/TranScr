"""Exercise real provider parsers and HTTP flows without contacting services."""

import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs

import httpx

from modules.translation.argos import ArgosTranslateProvider, sentence_parts
from modules.translation.batching import TranslationCancelled
from modules.translation.google import GoogleTranslateProvider
from modules.translation.yandex import YandexTranslateProvider


class HTTPProviderTests(unittest.TestCase):
    def provider(self, cls, handler, **kwargs):
        provider = cls(**kwargs)
        provider._client.close()
        provider._client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(provider.close)
        return provider

    def test_google_batches_without_merging_regions(self):
        requests = []

        def handler(request):
            requests.append(parse_qs(request.content.decode())["q"])
            return httpx.Response(200, json=["Первый", "Второй"])

        provider = self.provider(GoogleTranslateProvider, handler)
        self.assertEqual(
            provider.translate_many(["First", "Second"]), ["Первый", "Второй"]
        )
        self.assertEqual(requests, [["First", "Second"]])

    def test_google_retries_only_damaged_item(self):
        requests = []

        def handler(request):
            values = parse_qs(request.content.decode())["q"]
            requests.append(values)
            if request.url.host == "clients5.google.com":
                return httpx.Response(200, json=["Первый", None, "Третий"])
            return httpx.Response(200, json=[[["Второй", "Second"]]])

        provider = self.provider(GoogleTranslateProvider, handler)
        self.assertEqual(
            provider.translate_many(["First", "Second", "Third"]),
            ["Первый", "Второй", "Третий"],
        )
        self.assertEqual(requests, [["First", "Second", "Third"], ["Second"]])

    def test_google_wrong_batch_count_falls_back_without_crossing_sources(self):
        def handler(request):
            if request.url.host == "clients5.google.com":
                return httpx.Response(200, json=["wrong count"])
            value = parse_qs(request.content.decode())["q"][0]
            return httpx.Response(200, json=[[[value + " translated", value]]])

        provider = self.provider(GoogleTranslateProvider, handler)
        self.assertEqual(
            provider.translate_many(["First", "Second"]),
            ["First translated", "Second translated"],
        )

    def test_google_auto_language_shape_and_partial_error(self):
        values = GoogleTranslateProvider._parse_batch([["Привет", "en"], None], 2)
        self.assertEqual(values[0], "Привет")
        self.assertIsInstance(values[1], ValueError)
        with self.assertRaises(ValueError):
            GoogleTranslateProvider._parse([[[None]]])

    def test_google_unavailable_addresses_are_not_hammered_next_capture(self):
        requests = []

        def handler(request):
            requests.append(request.url.host)
            return httpx.Response(429, headers={"Retry-After": "60"})

        provider = self.provider(GoogleTranslateProvider, handler)
        self.assertIsInstance(provider.translate_many(["First"])[0], Exception)
        self.assertIsInstance(provider.translate_many(["Second"])[0], Exception)
        self.assertEqual(len(requests), 2)

    def test_yandex_api_key_does_not_require_folder(self):
        bodies = []

        def handler(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200, json={"translations": [{"text": "Первый"}, {"text": "Второй"}]}
            )

        provider = self.provider(YandexTranslateProvider, handler)
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test-key", "YC_FOLDER_ID": ""}):
            self.assertEqual(
                provider.translate_many(["First", "Second"]), ["Первый", "Второй"]
            )
        self.assertNotIn("folderId", bodies[0])
        self.assertEqual(bodies[0]["sourceLanguageCode"], "en")

    def test_yandex_batches_character_limit_and_keeps_order(self):
        def handler(request):
            body = json.loads(request.content)
            self.assertLessEqual(sum(map(len, body["texts"])), 10000)
            return httpx.Response(
                200,
                json={
                    "translations": [{"text": text.upper()} for text in body["texts"]]
                },
            )

        provider = self.provider(YandexTranslateProvider, handler)
        texts = ["a" * 6000, "b" * 6000]
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test-key"}):
            self.assertEqual(
                provider.translate_many(texts), [text.upper() for text in texts]
            )

    def test_yandex_partial_reply_keeps_other_items(self):
        provider = self.provider(
            YandexTranslateProvider,
            lambda request: httpx.Response(
                200, json={"translations": [{"text": "Первый"}, {}]}
            ),
        )
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test-key"}):
            values = provider.translate_many(["First", "Second"])
        self.assertEqual(values[0], "Первый")
        self.assertIsInstance(values[1], ValueError)

    def test_yandex_shared_cooldown_and_missing_credentials(self):
        handler = Mock(return_value=httpx.Response(503))
        provider = self.provider(YandexTranslateProvider, handler)
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test-key"}):
            self.assertIsInstance(provider.translate_many(["First"])[0], Exception)
            self.assertIsInstance(provider.translate_many(["Second"])[0], Exception)
        self.assertEqual(handler.call_count, 1)
        provider = self.provider(YandexTranslateProvider, handler)
        with patch.dict(os.environ, {"YANDEX_API_KEY": ""}):
            self.assertIsInstance(provider.translate_many(["Third"])[0], RuntimeError)
        self.assertEqual(handler.call_count, 1)

    def test_network_close_is_idempotent_and_cannot_reopen(self):
        for cls in (GoogleTranslateProvider, YandexTranslateProvider):
            provider = self.provider(cls, lambda request: httpx.Response(500))
            provider.close()
            provider.close()
            with self.assertRaises(RuntimeError):
                provider.translate_many(["Hello"])

    def test_streaming_response_checks_cancel_between_chunks(self):
        cancel = Event()

        class Stream(httpx.SyncByteStream):
            def __iter__(self):
                yield b"["
                cancel.set()
                yield b'"translation"]'

        provider = self.provider(
            GoogleTranslateProvider,
            lambda request: httpx.Response(200, stream=Stream()),
        )
        self.assertIsInstance(
            provider.translate_many(["First"], cancel=cancel)[0], TranslationCancelled
        )

    def test_oversized_service_response_is_closed(self):
        class Stream(httpx.SyncByteStream):
            closed = False

            def __iter__(self):
                yield b"x" * 100

            def close(self):
                self.closed = True

        stream = Stream()
        response = httpx.Response(200, stream=stream)
        provider = self.provider(YandexTranslateProvider, lambda request: response)
        provider.MAX_RESPONSE_BYTES = 10
        with patch.dict(os.environ, {"YANDEX_API_KEY": "test-key"}):
            result = provider.translate_many(["First"])[0]
        self.assertIsInstance(result, RuntimeError)
        self.assertTrue(response.is_closed)
        self.assertTrue(stream.closed)

    def test_close_cancels_queued_network_work_without_waiting_for_response(self):
        entered, release = Event(), Event()

        def handler(request):
            entered.set()
            release.wait(2)
            return httpx.Response(200, json=["Готово"])

        provider = self.provider(GoogleTranslateProvider, handler)
        with ThreadPoolExecutor(2) as pool:
            try:
                pending = pool.submit(provider.translate_many, ["First"])
                self.assertTrue(entered.wait(1))
                closing = pool.submit(provider.close)
                self.assertIsInstance(
                    pending.result(timeout=0.7)[0], TranslationCancelled
                )
            finally:
                release.set()
            closing.result(timeout=1)


class ArgosTests(unittest.TestCase):
    def provider(self, reply=None):
        provider = ArgosTranslateProvider()
        provider._beam_size = 2
        provider._package = SimpleNamespace(
            tokenizer=SimpleNamespace(
                encode=lambda text: [text], decode=lambda tokens: " ".join(tokens)
            ),
            target_prefix="__ru__",
        )
        provider._engine = Mock()
        provider._engine.translate_batch.return_value = reply or [
            SimpleNamespace(hypotheses=[["__ru__Первая."]]),
            SimpleNamespace(hypotheses=[["__ru__Вторая."]]),
            SimpleNamespace(hypotheses=[["__ru__Третья."]]),
        ]
        self.addCleanup(provider.close)
        return provider

    def test_sentence_mapping_uses_one_native_batch(self):
        provider = self.provider()
        self.assertEqual(
            provider.translate_many(["First. Second.", "Third."]),
            ["Первая.\nВторая.", "Третья."],
        )
        self.assertEqual(provider._engine.translate_batch.call_count, 1)
        options = provider._engine.translate_batch.call_args.kwargs
        self.assertFalse(options["return_scores"])
        self.assertEqual(options["beam_size"], 2)
        self.assertEqual(options["max_input_length"], 0)

    def test_abbreviations_identifiers_and_decimals_remain_intact(self):
        text = "Dr. Smith served in the U.S. Navy. No. 6 Squadron flew A-4s. It cost 3.14 dollars. A. Brown left."
        parts = sentence_parts([text])
        self.assertEqual(
            [part.text for part in parts],
            [
                "Dr. Smith served in the U.S. Navy.",
                "No. 6 Squadron flew A-4s.",
                "It cost 3.14 dollars.",
                "A. Brown left.",
            ],
        )
        self.assertEqual(" ".join(part.text for part in parts), text)
        self.assertEqual(len(sentence_parts([text], "de")), 1)

    def test_long_sentences_keep_native_input_bounded(self):
        parts = sentence_parts(["long " * 900])
        self.assertTrue(all(len(part.text) <= 1000 for part in parts))
        self.assertEqual(sum(len(part.text.split()) for part in parts), 900)

    def test_one_bad_native_result_does_not_erase_other_regions(self):
        provider = self.provider(
            [
                SimpleNamespace(hypotheses=[]),
                SimpleNamespace(hypotheses=[["__ru__Второй"]]),
            ]
        )
        values = provider.translate_many(["First", "Second"])
        self.assertIsInstance(values[0], ValueError)
        self.assertEqual(values[1], "Второй")

    def test_close_unloads_once_and_prevents_reopen(self):
        provider = self.provider()
        engine = provider._engine
        provider.close()
        provider.close()
        engine.unload_model.assert_called_once()
        with self.assertRaises(RuntimeError):
            provider.translate_many(["First"])
        with self.assertRaises(RuntimeError):
            provider.warmup()

    def test_cancel_while_waiting_for_native_engine(self):
        provider = self.provider()
        cancel = Event()
        with provider._lock, ThreadPoolExecutor(1) as pool:
            pending = pool.submit(provider.translate_many, ["First"], cancel=cancel)
            cancel.set()
            self.assertIsInstance(pending.result(timeout=0.7)[0], TranslationCancelled)
        provider._engine.translate_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
