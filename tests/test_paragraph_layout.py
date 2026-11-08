import unittest

from modules.ocr.types import OCRText
from modules.text_processing.text_regions import (
    group_ocr_texts,
    layout_slots,
    split_translation_to_source_lines,
)


class ParagraphTests(unittest.TestCase):
    def test_indented_justified_paragraph(self):
        lines = [
            OCRText("The first row", 20, 0, 300, 15),
            OCRText("second row", 0, 18, 300, 33),
            OCRText("third row", 0, 36, 300, 51),
            OCRText("end", 0, 54, 30, 69),
        ]
        regions = group_ocr_texts(lines)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].alignment, "justify")
        self.assertEqual(layout_slots(regions[0])[-1].right, 30)
        translated = "первая вторая третья четвертая пятая шестая седьмая восьмая"
        layout = split_translation_to_source_lines(
            regions[0], translated, lambda text: len(text) * 5
        )
        self.assertEqual(" ".join(text for _, text in layout), translated)

    def test_small_overlap_is_split_without_losing_rows(self):
        regions = group_ocr_texts(
            [OCRText("first", 0, 0, 100, 20), OCRText("second", 0, 18, 100, 38)]
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(len(regions[0].lines), 2)
        self.assertEqual(regions[0].lines[0].bottom, regions[0].lines[1].top)

    def test_horizontal_fragments_never_join(self):
        regions = group_ocr_texts(
            [
                OCRText("The", 0, 0, 20, 15),
                OCRText("aircraft", 30, 1, 100, 15),
                OCRText("Other column", 200, 0, 300, 15),
            ]
        )
        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[0].text, "The")

    def test_short_sentence_end_stops_paragraph_flow(self):
        lines = [
            OCRText("A long first line", 0, 0, 300, 15),
            OCRText("ends today.", 0, 18, 210, 33),
            OCRText("The next paragraph starts here", 0, 36, 300, 51),
        ]
        regions = group_ocr_texts(lines)
        self.assertEqual(len(regions), 2)
        self.assertEqual(len(regions[0].lines), 2)

    def test_real_magazine_ocr_keeps_four_body_paragraphs(self):
        import json
        from pathlib import Path

        data = json.loads(
            (Path(__file__).parent / "fixtures/magazine_ocr.json").read_text(
                encoding="utf-8"
            )
        )
        regions = group_ocr_texts([OCRText(**item) for item in data])
        self.assertEqual([len(region.lines) for region in regions], [4, 3, 4, 6])
        self.assertTrue(regions[0].text.endswith("today."))
        self.assertTrue(regions[3].text.startswith("The system."))
        for region in regions:
            self.assertTrue(
                all(line.right < 390 for line in region.lines)
                or all(line.left > 390 for line in region.lines)
            )

    def test_source_text_preserved(self):
        regions = group_ocr_texts(
            [
                OCRText("An instru-", 0, 0, 100, 15),
                OCRText("ment panel", 0, 18, 100, 33),
            ]
        )
        self.assertEqual(regions[0].text, "An instru-\nment panel")
