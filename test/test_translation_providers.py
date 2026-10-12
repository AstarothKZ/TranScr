"""Offline tests for translation helpers and provider request handling."""

import os
import unittest
from unittest.mock import Mock, patch

from modules.translation.text_chunks import split_text
from modules.translation.mymemory import MyMemoryTranslateProvider
from modules.translation.baidu import BaiduTranslateProvider


class TextChunkTests(unittest.TestCase):
    def test_split_text_obeys_limit(self):
        chunks = split_text("alpha beta gamma\n" + ("word " * 150), 100)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 100 for chunk in chunks))
        self.assertIn("alpha beta gamma", chunks[0])

    def test_long_word_is_split_without_dropping_text(self):
        source = "x" * 120
        chunks = split_text(source, 50)
        self.assertEqual("".join(chunks), source)
        self.assertTrue(all(len(chunk) <= 50 for chunk in chunks))


class MyMemoryProviderTests(unittest.TestCase):
    def test_uses_https_api_and_target_russian(self):
        provider = MyMemoryTranslateProvider()
        response = Mock()
        response.json.return_value = {
            "responseStatus": 200,
            "responseData": {"translatedText": "Крылья славы"},
        }
        response.raise_for_status.return_value = None
        try:
            with patch.object(provider._client, "get", return_value=response) as get:
                result = provider.translate("WINGS OF FAME")
            self.assertEqual(result, "Крылья славы")
            get.assert_called_once()
            self.assertEqual(get.call_args.args[0], "https://api.mymemory.translated.net/get")
            self.assertEqual(get.call_args.kwargs["params"]["langpair"], "en|ru")
        finally:
            provider.close()

    def test_numeric_only_chunk_is_not_sent_to_service(self):
        provider = MyMemoryTranslateProvider()
        try:
            with patch.object(provider._client, "get") as get:
                result = provider.translate("145063, 145071")
            self.assertEqual(result, "145063, 145071")
            get.assert_not_called()
        finally:
            provider.close()

    def test_api_error_is_reported(self):
        provider = MyMemoryTranslateProvider()
        response = Mock()
        response.json.return_value = {
            "responseStatus": 429,
            "responseDetails": "Too Many Requests",
        }
        response.raise_for_status.return_value = None
        try:
            with patch.object(provider._client, "get", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "Too Many Requests"):
                    provider.translate("Hello World")
        finally:
            provider.close()


class BaiduProviderTests(unittest.TestCase):
    def test_missing_credentials_has_clear_error(self):
        provider = BaiduTranslateProvider()
        try:
            with patch.dict(os.environ, {"BAIDU_APP_ID": "", "BAIDU_API_KEY": ""}):
                with self.assertRaisesRegex(RuntimeError, "BAIDU_APP_ID"):
                    provider.translate("Hello")
        finally:
            provider.close()

    def test_parses_baidu_api_response(self):
        provider = BaiduTranslateProvider()
        response = Mock()
        response.json.return_value = {"trans_result": [{"src": "Hello", "dst": "Привет"}]}
        response.raise_for_status.return_value = None
        try:
            with (
                patch.dict(os.environ, {"BAIDU_APP_ID": "demo-id", "BAIDU_API_KEY": "demo-key"}),
                patch.object(provider._client, "post", return_value=response) as post,
                patch.object(provider, "MIN_REQUEST_INTERVAL", 0),
            ):
                result = provider.translate("Hello")
            self.assertEqual(result, "Привет")
            payload = post.call_args.kwargs["data"]
            self.assertEqual(payload["from"], "en")
            self.assertEqual(payload["to"], "ru")
            self.assertIn("sign", payload)
        finally:
            provider.close()


if __name__ == "__main__":
    unittest.main()
