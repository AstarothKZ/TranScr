"""Unit tests for the translation router without internet access."""

import unittest
from unittest.mock import patch

from modules.translation.translator import Translator


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.translator = Translator(from_code="en", to_code="ru")

    def test_google_is_default(self):
        self.assertEqual(self.translator.provider, "google")

    def test_default_provider_uses_google(self):
        with patch.object(self.translator, "_google") as google:
            google.return_value.translate.return_value = "Привет"
            result = self.translator.translate_text("Hello")
        self.assertEqual(result, "Привет")
        google.return_value.translate.assert_called_once_with("Hello")

    def test_yandex_can_be_selected_per_call(self):
        with patch.object(self.translator, "_yandex") as yandex:
            yandex.return_value.translate.return_value = "Мир"
            result = self.translator.translate_text("World", provider="yandex")
        self.assertEqual(result, "Мир")
        yandex.return_value.translate.assert_called_once_with("World")

    def test_hybrid_uses_argos_if_google_fails(self):
        with (
            patch.object(self.translator, "_google") as google,
            patch.object(self.translator, "_argos") as argos,
        ):
            google.return_value.translate.side_effect = OSError("offline")
            argos.return_value.translate.return_value = "Привет"
            result = self.translator.translate_text("Hello", provider="hybrid")
        self.assertEqual(result, "Привет")
        argos.return_value.translate.assert_called_once_with("Hello")

    def test_empty_text_does_not_call_provider(self):
        with patch.object(self.translator, "_google") as google:
            self.assertEqual(self.translator.translate_text("   \n  "), "")
        google.assert_not_called()

    def test_repeated_text_uses_cache(self):
        with patch.object(self.translator, "_google") as google:
            google.return_value.translate.return_value = "Привет"
            self.assertEqual(self.translator.translate_text("Hello"), "Привет")
            self.assertEqual(self.translator.translate_text("Hello"), "Привет")
        google.return_value.translate.assert_called_once_with("Hello")

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            self.translator.translate_text("Hello", provider="unknown")


if __name__ == "__main__":
    unittest.main()
