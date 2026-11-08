import unittest

from modules.text_processing.filters import should_translate, untranslated_fragments


class ResolutionFilterTests(unittest.TestCase):
    def test_resolution_and_pc_logo_are_not_translated(self):
        for value in ('1080 x 1523', '1080×1523', '1080 х 1523', '1920 X 1080 px', 'PC'):
            with self.subTest(value=value):
                self.assertFalse(should_translate(value))
        self.assertTrue(should_translate('Connect your PC'))
        self.assertTrue(should_translate('Image size is 1080 x 1523'))

class FilterTests(unittest.TestCase):
    def test_sizes_and_transfer_rates_are_not_sent(self):
        for text in (
            "205.3 K6",
            "205.3 Кb",
            "205.3 Kб",
            "205,3 К Б",
            "B 205.3 KB",
            "В 205,3 Кб.",
            "205.3 КБ",
            "205,3 кб",
            "1 024 Б",
            "20 MB",
            "12 MiB",
            "5 КБ/с",
            "120 bytes",
            "1.5 GB/s",
        ):
            with self.subTest(text=text):
                self.assertFalse(should_translate(text))

    def test_size_inside_sentence_is_preserved(self):
        self.assertTrue(should_translate("Download the 205.3 KB file"))
        self.assertTrue(should_translate("File size: 205.3 KB"))

    def test_standalone_ui_labels(self):
        for text in ("PNG", "РУС", "PYC", "RUS", "ENG", "JPEG"):
            self.assertFalse(should_translate(text), text)
        for text in ("Save as PNG", "The ENG button", "NASA", "A 6 mm screw"):
            self.assertTrue(should_translate(text), text)

    def test_partial_translation_and_proper_names(self):
        self.assertEqual(untranslated_fragments("является very limited"), [(9, 21)])
        self.assertEqual(untranslated_fragments("Система Saab RGS и NASA"), [])
        self.assertEqual(untranslated_fragments("NASA"), [])
