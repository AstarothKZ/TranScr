"""Unit tests for Google response parsing and the unchanged-text guard."""

import unittest
from unittest.mock import Mock

from modules.translation.google import GoogleTranslateProvider


class GoogleProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = GoogleTranslateProvider.__new__(GoogleTranslateProvider)
        self.provider.from_code = "en"
        self.provider.to_code = "ru"

    def test_rejects_long_text_returned_unchanged(self):
        text = "This paragraph should be translated into Russian"
        with self.assertRaisesRegex(RuntimeError, "исходный текст без перевода"):
            self.provider._check_translation(text, text)

    def test_allows_short_unchanged_labels(self):
        self.provider._check_translation("A-4", "A-4")

    def test_allows_text_that_was_translated(self):
        self.provider._check_translation("Hello world, welcome here", "Привет мир, добро пожаловать")

    def test_client_settings_are_not_hardcoded_into_translator_router(self):
        provider = GoogleTranslateProvider.__new__(GoogleTranslateProvider)
        provider._client = Mock()
        provider.from_code = "en"
        provider.to_code = "ru"
        provider._check_translation = Mock()
        provider._client.get.return_value.raise_for_status.return_value = None
        provider._client.get.return_value.json.return_value = [[ ["Привет мир", "Hello world", None, None] ]]
        self.assertEqual(provider.translate("Hello world"), "Привет мир")
        provider._client.get.assert_called_once()
        self.assertEqual(provider._client.get.call_args.kwargs["params"]["tl"], "ru")


if __name__ == "__main__":
    unittest.main()
