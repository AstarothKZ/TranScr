"""Contracts for exact source placement, independent of translation quality."""
import unittest
import numpy as np

from modules.ocr.labels import recover_labels
from modules.ocr.types import OCRText
from modules.overlay.fonts import PixelFont
from modules.overlay.layout import fit_region
from modules.overlay.rendering import render_row
from modules.text_processing.filters import should_translate
from modules.text_processing.text_regions import group_ocr_texts


class RevisionLayoutTests(unittest.TestCase):
    def test_word_dates_and_three_letter_words_are_translated(self):
        for text in ('November 5, 2026', 'Monday, 5 November', 'May', 'Sun',
                     'YES', 'The', 'Kraków', 'Łódź', '東京駅', 'rhythms'):
            self.assertTrue(should_translate(text), text)
        for text in ('05.11.2026', '12:30', 'HP', 'MP', 'A', '12'):
            self.assertFalse(should_translate(text), text)

    def test_two_line_heading_does_not_justify(self):
        lines = [OCRText('Back to Night City —', 87, 200, 295, 220),
                 OCRText('Cyberpunk: Edgerunners 2', 87, 224, 363, 244)]
        region, = group_ocr_texts(lines)
        self.assertEqual(region.alignment, 'left')
        plan = fit_region(region, 'Назад в Найт-Сити — Киберпанк: Бегущие по краю 2', PixelFont)
        self.assertEqual([r.source for r in plan], lines)
        self.assertTrue(all(len(r.runs) <= 1 for r in plan))
        for row in plan:
            render_row(row, 'white')  # Validates actual ink, not advance widths.

    def test_missing_interior_words_do_not_expand_the_source(self):
        lines = [OCRText('body row', 10, y, 310, y+16) for y in range(0, 120, 20)]
        lines[3] = OCRText('incomplete', 10, 60, 265, 76)
        regions = group_ocr_texts(lines)
        self.assertEqual([l for r in regions for l in r.lines], lines)

    def test_justification_fills_source_width(self):
        lines = [OCRText('a body row', 0, y, 300, y+20) for y in (0, 24, 48)]
        region, = group_ocr_texts(lines)
        plan = fit_region(region, 'один два три четыре пять шесть', PixelFont)
        self.assertTrue(plan)
        for row in plan:
            for first, second in zip(row.runs, row.runs[1:]):
                gap = second[0] - first[0] - row.font.measure(first[2])
                self.assertGreaterEqual(gap, row.font.measure(' '))
            render_row(row, 'black')

    def test_partial_label_recovery_retains_original(self):
        image = np.full((30, 400, 3), 20, dtype=np.uint8)
        for x in (5, 65, 125, 185, 245, 305):
            image[8:20, x:x+30] = 240
        original = OCRText('merged footer links', 3, 5, 338, 23)
        calls = []
        def recognize(crop):
            calls.append(1)
            return [] if len(calls) == 2 else [OCRText('Support', 2, 2, 50, 20)]
        self.assertEqual(recover_labels(image, [original], recognize), [original])

