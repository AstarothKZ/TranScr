"""Source-selection contracts: preserve data and translate meaningful captions."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from modules.ocr.types import OCRText
from modules.text_processing.filters import (
    exclusion_reason,
    is_contact_address,
    should_translate,
    table_identifier_indices,
    translation_changed,
    untranslated_fragments,
)
from modules.text_processing.text_regions import group_ocr_texts


class TextFilterTests(unittest.TestCase):
    def check_cases(self, values, expected):
        for value in values:
            with self.subTest(text=value):
                self.assertEqual(should_translate(value), expected)

    def test_up_to_two_alphanumeric_characters_are_skipped(self):
        self.check_cases(
            ["A", "AA", "АА", "A A", "(AA)", "A-4", "Go", "OK", "No", "c/n", "12"],
            False,
        )

    def test_three_letter_words_are_eligible_in_any_case(self):
        self.check_cases(
            [
                "Yes",
                "YES",
                "Buy",
                "BUY",
                "New",
                "The",
                "not",
                "For",
                "Was",
                "Eye",
                "You",
            ],
            True,
        )

    def test_codes_versions_and_serials_remain_unchanged(self):
        self.check_cases(
            [
                "AGM-65A",
                "A4-C",
                "A-4C",
                "A‑4C",
                "A—4C",
                "A4C",
                "7385-A",
                "3247dgh",
                "v1.2.3",
                "1.2.3-beta",
                "2026.11.14",
                "12345 67890",
                "145108 12354 A-4C",
                "AGM-65A A4-C",
                "(AGM-65A)",
                "AGM-65A,",
                "Ａ４－Ｃ",
            ],
            False,
        )

    def test_codes_within_prose_are_not_removed(self):
        self.check_cases(
            [
                "The AGM-65A missile",
                "This is an A4-C variant",
                "Version 1.2.3",
                "3D printing",
                "Known Republic of Singapore Air Force TA4S-1/TA-4SU",
                "BuNo identifies the aircraft",
                "Explain c/n here",
                "1000 lb (454 kg)",
                "In service 2003",
                "AJT Detachment, Cazaux 04",
            ],
            True,
        )

    def test_known_technical_fields_are_not_guessed_as_words(self):
        self.check_cases(
            ["BuNo", "BUNO", "buno", "BuNo.", "(BuNo)", "p/n", "s/n"], False
        )
        self.check_cases(
            ["Units", "Fate/notes", "Fate", "Notes", "Business", "Contact"], True
        )

    def test_fully_written_month_dates_translate(self):
        self.check_cases(
            [
                "November 5, 2026",
                "November 6, 2026",
                "5 November 2026",
                "5 NOVEMBER 2026",
                "November 5",
                "May 5",
                "5 May 2026",
                "12 April 1984",
                "August 28, 1984",
                "5 ноября 2026",
            ],
            True,
        )

    def test_numeric_and_abbreviated_dates_remain_data(self):
        self.check_cases(
            [
                "12/04/1984",
                "2026-11-14",
                "14.11.2026",
                "12:34:56",
                "Nov. 5, 2026",
                "5 Nov 2026",
                "Apr 12, 1984",
            ],
            False,
        )
        self.check_cases(
            ["Deliveries took place in Nov 1984", "January deliveries"], True
        )

    def test_addresses_with_punctuation_ports_and_query_strings(self):
        self.check_cases(
            [
                "support@example.com",
                "support@example.com,",
                "(support@example.com).",
                "https://example.com",
                "https://example.com:443/path?q=hello#item",
                "https://localhost:8080/page",
                "https://[::1]:8080/",
                "example.com:443",
                "example.CoM",
                "example.com?lang=en",
                "example.com#contact",
                "www.example.com/path",
                "cdprojektred.com/product-safety",
                "mailto:support@example.com",
                "ftp://example.com/file",
                "<support@example.com>",
                "[https://example.com]",
            ],
            False,
        )

    def test_multiple_addresses_or_data_atoms_are_skipped_together(self):
        self.check_cases(
            [
                "support@example.com jobs@example.com",
                "https://example.com\nhttps://other.org",
                "AGM-65A support@example.com 123456",
            ],
            False,
        )

    def test_addresses_within_prose_remain_part_of_translation(self):
        self.check_cases(
            [
                "Write to support@example.com",
                "Visit https://example.com:443/path",
                "Business support@example.com",
                "Our locations",
                "Press & Content Creators",
            ],
            True,
        )
        self.assertFalse(is_contact_address("Write to support@example.com"))
        self.assertFalse(is_contact_address("https://"))
        self.assertFalse(is_contact_address("https://example.com:bad"))
        self.assertTrue(should_translate("BK.Black Kn.ts"))
        self.assertTrue(is_contact_address("https://example.CoM"))

    def test_vowel_and_consonant_noise_is_case_insensitive(self):
        self.check_cases(
            [
                "aeiou",
                "AEIOU",
                "AeIoU",
                "aaa",
                "a a a",
                "AE IOU",
                "qwrtt",
                "QWRTT",
                "bcd",
                "TRH P s",
                "m d rk c",
                "zzz",
                "YYY",
            ],
            False,
        )

    def test_y_can_carry_a_vowel_and_noise_does_not_strip_prose(self):
        self.check_cases(
            [
                "myths",
                "rhythms",
                "crypt",
                "glyphs",
                "eye",
                "HEY",
                "The code is QWRTT",
                "Explain AEIOU here",
                "Black Knights No.4",
            ],
            True,
        )

    def test_numeric_units_dimensions_and_ui_labels(self):
        self.check_cases(
            [
                "1000",
                "1,000.00",
                "42%",
                "$100",
                "142, 145",
                "№ 1234",
                "No. 1234",
                "1920x1080",
                "1920 × 1080 px",
                "12 MB",
                "12 MiB/s",
                "PNG",
                "PDF",
                "ENG",
                "screenshot.png",
                "screenshot.png | 12 MB",
            ],
            False,
        )
        self.check_cases(
            [
                "PNG image settings",
                "PDF document",
                "File size is 12 MB",
                "1920 pixels wide",
            ],
            True,
        )

    def test_empty_and_non_text_have_explainable_reasons(self):
        for value in (None, "", " \n ", "***", "─┼─"):
            with self.subTest(text=value):
                self.assertFalse(should_translate(value))
                self.assertIsNotNone(exclusion_reason(value))
        self.assertEqual(exclusion_reason("AA"), "short")
        self.assertEqual(exclusion_reason("BuNo"), "technical_label")
        self.assertEqual(exclusion_reason("A4-C"), "identifier")
        self.assertEqual(exclusion_reason("support@example.com,"), "contact")
        self.assertIsNone(exclusion_reason("November 5, 2026"))

    def test_table_data_is_retained_as_geometry_context(self):
        lines = [
            OCRText("145108", 0, 20, 40, 30),
            OCRText("12354", 50, 20, 90, 30),
            OCRText("A4-C", 100, 20, 130, 30),
            OCRText("In service 2001", 150, 20, 270, 30),
            OCRText("147841", 0, 32, 40, 42),
            OCRText("12605", 50, 32, 90, 42),
            OCRText("A4-C", 100, 32, 130, 42),
            OCRText("In service 2003", 150, 32, 270, 42),
        ]
        eligible = [line for line in lines if should_translate(line.text)]
        regions = group_ocr_texts(eligible, context=lines)
        self.assertEqual(
            [region.text for region in regions], ["In service 2001", "In service 2003"]
        )

    def test_table_columns_preserve_misread_ids_without_hiding_status_words(self):
        lines = []
        for row, code in enumerate(["A4-C", "AAC", "A4-C"]):
            y = 20 + row * 12
            lines.extend(
                [
                    OCRText("145108", 0, y, 40, y + 10),
                    OCRText("12354", 50, y, 90, y + 10),
                    OCRText(code, 100, y, 130, y + 10),
                    OCRText("In service 2001", 150, y, 270, y + 10),
                ]
            )
        self.assertEqual(table_identifier_indices(lines), {6})
        # Two-digit data is a geometric anchor too, despite the short-text rule.
        short_ids = [
            OCRText(
                "12" if line.text in {"145108", "12354"} else line.text,
                line.left,
                line.top,
                line.right,
                line.bottom,
            )
            for line in lines
        ]
        self.assertEqual(table_identifier_indices(short_ids), {6})
        # The same OCR token outside a table is not classified by geometry.
        self.assertEqual(table_identifier_indices([lines[6]]), set())

    def test_table_headers_and_standalone_short_buttons_are_not_data_cells(self):
        lines = [OCRText("Buy", 100, 0, 130, 10)]
        for row in range(3):
            y = 20 + row * 12
            lines.extend(
                [
                    OCRText("145108", 0, y, 40, y + 10),
                    OCRText("12354", 50, y, 90, y + 10),
                    OCRText("A4-C", 100, y, 130, y + 10),
                ]
            )
        self.assertNotIn(0, table_identifier_indices(lines))

    def test_translation_comparison_and_partial_detection_are_preserved(self):
        self.assertFalse(translation_changed("Open", "OPEN!"))
        self.assertFalse(translation_changed("1000 lb", "1000lb"))
        self.assertTrue(translation_changed("Open", "Открыть"))
        self.assertEqual(untranslated_fragments("Текст Windows OCR"), [])
        self.assertTrue(untranslated_fragments("Текст with untranslated words"))

    def test_real_pipeline_sends_words_and_dates_but_not_data_cells(self):
        from core.pipeline import process_capture

        lines = [OCRText("BuNo", 0, 0, 40, 10)]
        for row, code in enumerate(["A4-C", "AAC", "A4-C"]):
            y = 20 + row * 12
            lines.extend(
                [
                    OCRText("145108", 0, y, 40, y + 10),
                    OCRText("12354", 50, y, 90, y + 10),
                    OCRText(code, 100, y, 130, y + 10),
                    OCRText("In service 2001", 150, y, 270, y + 10),
                ]
            )
        lines.extend(
            [
                OCRText("November 5, 2026", 0, 70, 150, 80),
                OCRText("AA", 160, 70, 180, 80),
            ]
        )
        queue = Queue()

        def submit(image, token):
            queue.put(("result", lines, 1, token))
            return True

        translator = Mock()
        translator.translate_many.side_effect = lambda texts, cancel: {
            text: "Перевод" for text in texts
        }
        translator.last_stats = None
        capture = Mock()
        capture.capture_region.return_value = (
            np.zeros((100, 300, 3), dtype=np.uint8),
            None,
        )
        overlay = SimpleNamespace(background_ms=0, layout_ms=0, raster_ms=0)
        with ThreadPoolExecutor(1) as executor:
            runtime = SimpleNamespace(
                ocr_worker=SimpleNamespace(submit=submit),
                ocr_results=queue,
                translator=translator,
                translation_executor=executor,
            )
            with (
                patch("core.pipeline.has_source_ink", return_value=True),
                patch(
                    "core.pipeline.prepare_lines", side_effect=lambda image, raw: raw
                ),
                patch("core.pipeline.prepare_background"),
                patch("core.pipeline.prepare_overlay", return_value=overlay),
            ):
                result = process_capture(capture, (0, 0, 300, 100), False, runtime)
        self.assertEqual(
            set(translator.translate_many.call_args.args[0]),
            {"In service 2001", "November 5, 2026"},
        )
        self.assertEqual(result.errors, [])


if __name__ == "__main__":
    unittest.main()
