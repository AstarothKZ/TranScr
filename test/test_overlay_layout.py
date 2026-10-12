"""Unit tests for grouping OCR lines into approximate overlay regions."""

import unittest

from modules.ocr.types import OCRText
from modules.overlay.text_blocks import group_ocr_texts


class OverlayLayoutTests(unittest.TestCase):
    def test_nearby_lines_are_grouped(self):
        items = [
            OCRText("Second line", 100, 120, 240, 134),
            OCRText("First line", 100, 101, 230, 115),
        ]
        regions = group_ocr_texts(items)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].text, "First line\nSecond line")
        self.assertEqual((regions[0].left, regions[0].top), (100, 101))
        self.assertEqual((regions[0].right, regions[0].bottom), (240, 134))

    def test_lines_with_large_gap_are_separate(self):
        items = [
            OCRText("Heading", 100, 100, 200, 114),
            OCRText("Paragraph", 100, 140, 280, 154),
        ]
        regions = group_ocr_texts(items)
        self.assertEqual(len(regions), 2)

    def test_far_apart_columns_are_not_grouped(self):
        items = [
            OCRText("Left column", 50, 100, 150, 114),
            OCRText("Right column", 600, 101, 730, 115),
        ]
        regions = group_ocr_texts(items)
        self.assertEqual(len(regions), 2)

    def test_empty_and_blank_items_are_ignored(self):
        self.assertEqual(group_ocr_texts([]), [])
        self.assertEqual(group_ocr_texts([OCRText("  ", 1, 1, 2, 2)]), [])

    def test_font_size_is_based_on_line_height_not_paragraph_height(self):
        items = [
            OCRText("Line one", 100, 100, 200, 114),
            OCRText("Line two", 100, 118, 200, 132),
            OCRText("Line three", 100, 136, 200, 150),
        ]
        region = group_ocr_texts(items)[0]
        self.assertLessEqual(region.font_size, 14)


if __name__ == "__main__":
    unittest.main()
