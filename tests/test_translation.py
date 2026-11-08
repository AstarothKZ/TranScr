import json
import threading
import unittest
from unittest.mock import Mock, patch

import httpx

from modules.translation.argos import ArgosTranslateProvider
from modules.translation.mymemory import MyMemoryTranslateProvider
from modules.translation.text_chunks import split_text
from modules.translation.translator import Translator
from modules.translation.yandex import YandexTranslateProvider


class TranslationTests(unittest.TestCase):
    def test_partial_translation_is_completed_before_caching(self):
        translator = self.translator()
        provider = Mock()
        provider.translate.side_effect = ["является very limited", "очень ограниченным"]
        translator._mymemory_provider = provider
        self.assertEqual(
            translator.translate_many(["is very limited"]),
            {"is very limited": "является очень ограниченным"},
        )
        self.assertEqual(provider.translate.call_args.args, ("very limited",))
        translator.translate_many(["is very limited"])
        self.assertEqual(provider.translate.call_count, 2)

    def test_partial_yandex_batch_is_completed(self):
        translator = self.translator(provider="yandex")
        provider = Mock()
        provider.translate_many.return_value = ["является very limited"]
        provider.translate.return_value = "очень ограниченным"
        translator._yandex_provider = provider
        self.assertEqual(
            translator.translate_many(["is very limited"])["is very limited"],
            "является очень ограниченным",
        )

    def test_failed_completion_uses_hybrid_fallback(self):
        translator = self.translator(hybrid_enabled=True)
        translator._mymemory_provider = Mock()
        translator._mymemory_provider.translate.side_effect = [
            "является very limited",
            "very limited",
        ]
        translator._argos_provider = Mock()
        translator._argos_provider.translate.return_value = "очень ограничено"
        self.assertEqual(
            translator.translate_many(["is very limited"])["is very limited"],
            "очень ограничено",
        )

    def test_failed_completion_is_not_cached(self):
        translator = self.translator()
        provider = Mock()
        provider.translate.side_effect = [
            "является very limited",
            "very limited",
            "очень ограничено",
        ]
        translator._mymemory_provider = provider
        self.assertEqual(
            translator.translate_many(["is very limited"])["is very limited"],
            "является very limited",
        )
        self.assertEqual(
            translator.translate_many(["is very limited"])["is very limited"],
            "очень ограничено",
        )

    def translator(self, **kwargs):
        instance = Translator(**kwargs)
        self.addCleanup(instance.close)
        return instance

    def test_yandex_batches_deduplicates_and_caches_regions(self):
        translator = self.translator(provider="yandex")
        provider = Mock()
        provider.translate_many.return_value = ["Один", "Два"]
        translator._yandex_provider = provider
        self.assertEqual(
            translator.translate_many(["One", "Two", "One"]),
            {"One": "Один", "Two": "Два"},
        )
        self.assertEqual(translator.translate_many(["Two"]), {"Two": "Два"})
        provider.translate_many.assert_called_once_with(["One", "Two"])

    def test_unchanged_text_does_not_run_fallback(self):
        translator = self.translator(hybrid_enabled=True)
        translator._mymemory_provider = Mock()
        translator._mymemory_provider.translate.return_value = "NASA"
        translator._argos_provider = Mock()
        self.assertEqual(translator.translate_many(["NASA"]), {"NASA": "NASA"})
        translator._argos_provider.translate.assert_not_called()

    def test_failed_yandex_batch_falls_back_without_individual_retries(self):
        translator = self.translator(provider="yandex", hybrid_enabled=True)
        translator._yandex_provider = Mock()
        translator._yandex_provider.translate_many.side_effect = RuntimeError("offline")
        translator._argos_provider = Mock()
        translator._argos_provider.translate.side_effect = lambda text: "ru:" + text
        result = translator.translate_many(["One", "Two"])
        self.assertEqual(result, {"One": "ru:One", "Two": "ru:Two"})
        translator._yandex_provider.translate.assert_not_called()

    def test_transport_failure_stops_remaining_frame_requests(self):
        translator = self.translator()
        translator._mymemory_provider = Mock()
        translator._mymemory_provider.translate.side_effect = httpx.ConnectError(
            "offline"
        )
        result = translator.translate_many([f"label {index}" for index in range(20)])
        self.assertEqual(len(result), 20)
        self.assertTrue(all(isinstance(value, Exception) for value in result.values()))
        self.assertLessEqual(translator._mymemory_provider.translate.call_count, 3)
        before = translator._mymemory_provider.translate.call_count
        translator.translate_many(["retry"])
        self.assertGreater(translator._mymemory_provider.translate.call_count, before)

    def test_transport_failure_with_successful_fallback_still_stops_primary(self):
        translator = self.translator(hybrid_enabled=True)
        translator._mymemory_provider = Mock()
        translator._mymemory_provider.translate.side_effect = httpx.ReadTimeout(
            "offline"
        )
        translator._argos_provider = Mock()
        translator._argos_provider.translate.return_value = "перевод"
        result = translator.translate_many([f"label {index}" for index in range(20)])
        self.assertTrue(all(value == "перевод" for value in result.values()))
        self.assertLessEqual(translator._mymemory_provider.translate.call_count, 3)

    def test_cancelled_frame_makes_no_requests(self):
        translator = self.translator()
        translator._mymemory_provider = Mock()
        cancel = threading.Event()
        cancel.set()
        self.assertEqual(translator.translate_many(["hello"], cancel), {})
        translator._mymemory_provider.translate.assert_not_called()

    def test_failure_is_not_cached(self):
        translator = self.translator()
        translator._mymemory_provider = Mock()
        translator._mymemory_provider.translate.side_effect = [
            RuntimeError("failure"),
            "привет",
        ]
        self.assertIsInstance(translator.translate_many(["hello"])["hello"], Exception)
        self.assertEqual(translator.translate_many(["hello"])["hello"], "привет")

    def test_partial_success_preserved(self):
        translator = self.translator()
        translator._mymemory_provider = Mock()

        def translate(text):
            if text == "bad":
                raise ValueError("bad response")
            return "перевод"

        translator._mymemory_provider.translate.side_effect = translate
        result = translator.translate_many(["good", "bad"])
        self.assertEqual(result["good"], "перевод")
        self.assertIsInstance(result["bad"], ValueError)


class ProviderTests(unittest.TestCase):
    def test_mymemory_rejects_null_text_and_invalid_payload(self):
        provider = MyMemoryTranslateProvider()
        provider._client.close()
        self.addCleanup(provider.close)
        for payload in ([], {"responseData": {"translatedText": None}}, {"responseData": ["bad"]}):
            with self.subTest(payload=payload):
                provider._client = httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=payload)
                ))
                try:
                    with self.assertRaises(RuntimeError):
                        provider.translate("Hello")
                finally:
                    provider._client.close()

    def test_argos_resolves_model_only_once(self):
        provider = ArgosTranslateProvider()
        model = Mock()
        model.translate.return_value = "перевод"
        provider._resolve_translation = Mock(return_value=model)
        provider.translate("one")
        provider.translate("two")
        provider._resolve_translation.assert_called_once()
        self.assertEqual(model.translate.call_count, 2)

    def provider(self, handler):
        provider = YandexTranslateProvider()
        provider._client.close()
        provider._client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(provider.close)
        return provider

    @patch.dict("os.environ", {"YANDEX_API_KEY": "test", "YC_FOLDER_ID": "test"})
    def test_yandex_preserves_order_and_respects_request_limit(self):
        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append(body["texts"])
            self.assertLessEqual(sum(map(len, body["texts"])), 10_000)
            return httpx.Response(
                200,
                json={
                    "translations": [{"text": value.upper()} for value in body["texts"]]
                },
            )

        provider = self.provider(handler)
        texts = ["first", "a" * 10_001, "last"]
        results = provider.translate_many(texts)
        self.assertEqual(results[0], "FIRST")
        self.assertEqual(results[1].replace("\n", ""), "A" * 10_001)
        self.assertEqual(results[2], "LAST")
        self.assertEqual(len(requests), 3)

    @patch.dict("os.environ", {"YANDEX_API_KEY": "test", "YC_FOLDER_ID": "test"})
    def test_yandex_rejects_missing_translations(self):
        provider = self.provider(
            lambda request: httpx.Response(200, json={"translations": []})
        )
        with self.assertRaises(RuntimeError):
            provider.translate_many(["hello"])

    def test_utf8_limit_preserves_multibyte_text(self):
        text = "Я" * 700 + "🙂" * 120
        chunks = split_text(text, 500, utf8_bytes=True)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode("utf-8")) <= 500 for chunk in chunks))

    def test_mymemory_uses_byte_limit(self):
        provider = MyMemoryTranslateProvider()
        self.addCleanup(provider.close)
        provider._translate_chunk = Mock(side_effect=lambda text: text)
        text = "Я" * 600
        self.assertEqual(provider.translate(text).replace("\n", ""), text)
        self.assertEqual(provider._translate_chunk.call_count, 3)

    def test_invalid_chunk_limit(self):
        with self.assertRaises(ValueError):
            split_text("hello", 0)


if __name__ == "__main__":
    unittest.main()
