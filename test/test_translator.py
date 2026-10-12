"""Unit-тесты маршрутизации переводчика без доступа к интернету."""

import unittest
from unittest.mock import patch

from modules.translation.translator import Translator


class TranslatorTests(unittest.TestCase):
    def setUp(self):
        self.translator = Translator(from_code="en", to_code="ru")

    def test_google_is_default(self):
        self.assertEqual(self.translator.provider, "google")

    def test_default_provider_uses_google(self):
        with patch.object(self.translator, "translate_google", return_value="Привет") as google:
            result = self.translator.translate_text("Hello")
        self.assertEqual(result, "Привет")
        google.assert_called_once_with("Hello")

    def test_yandex_can_be_selected_per_call(self):
        with patch.object(self.translator, "translate_yandex_cloud", return_value="Мир") as yandex:
            result = self.translator.translate_text("World", provider="yandex")
        self.assertEqual(result, "Мир")
        yandex.assert_called_once_with("World")

    def test_hybrid_uses_argos_if_google_fails(self):
        with (
            patch.object(self.translator, "translate_google", side_effect=OSError("offline")),
            patch.object(self.translator, "translate_argos", return_value="Привет") as argos,
        ):
            result = self.translator.translate_text("Hello", provider="hybrid")
        self.assertEqual(result, "Привет")
        argos.assert_called_once_with("Hello")

    def test_empty_text_does_not_call_provider(self):
        with patch.object(self.translator, "translate_google") as google:
            self.assertEqual(self.translator.translate_text("   \n  "), "")
        google.assert_not_called()

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(ValueError):
            self.translator.translate_text("Hello", provider="unknown")


if __name__ == "__main__":
    unittest.main()
