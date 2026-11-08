import unittest

from modules.ocr.types import OCRText
from modules.overlay.layout import fit_region
from modules.text_processing.text_regions import TextRegion, group_ocr_texts


class Font:
    def __init__(self, size):
        self.size = size

    def measure(self, text):
        return len(text) * self.size * 0.5

    def metrics(self, name):
        return self.size + 2


class LineLayoutTests(unittest.TestCase):
    def test_shorter_paragraph_uses_all_source_rows(self):
        region = self.region([240, 240, 240, 240])
        text = 'Один два три четыре пять шесть семь восемь'
        plan = fit_region(region, text, Font)
        self.assertTrue(all(row.text for row in plan))
        self.assertEqual(' '.join(row.text for row in plan), text)
        self.assert_inside(plan)

    def region(self, widths):
        lines = tuple(
            OCRText("source row", 0, i * 24, width, i * 24 + 20)
            for i, width in enumerate(widths)
        )
        return TextRegion("source", 0, 0, max(widths), lines[-1].bottom, 14, lines)

    def assert_inside(self, plan):
        for row in plan:
            for x, y, text in row.runs:
                self.assertGreaterEqual(x, row.source.left)
                self.assertGreaterEqual(y, row.source.top)
                self.assertLessEqual(
                    x + row.font.measure(text), row.source.right + 0.001
                )
                self.assertLessEqual(
                    y + row.font.metrics("linespace"), row.source.bottom + 0.001
                )

    def test_different_widths_preserved_and_no_words_lost(self):
        region = self.region([240, 240, 60])
        text = "Первое предложение затем второе предложение и короткий конец"
        plan = fit_region(region, text, Font)
        self.assertEqual([row.source for row in plan], list(region.lines))
        self.assertEqual(" ".join(row.text for row in plan), text)
        self.assertEqual(len({row.font.size for row in plan}), 1)
        self.assert_inside(plan)


    def test_justified_rows_reach_their_own_right_edge(self):
        plan = fit_region(
            self.region([200, 200, 100]),
            "один два три четыре пять шесть семь восемь девять",
            Font,
        )
        self.assertTrue(plan)
        for row in plan:
            if len(row.runs) > 1:
                x, _, word = row.runs[-1]
                self.assertAlmostEqual(x + row.font.measure(word), row.source.right - 1)
        self.assert_inside(plan)

    def test_short_translation_still_covers_all_source_rows(self):
        plan = fit_region(self.region([200, 200, 100]), "коротко", Font)
        self.assertEqual(len(plan), 3)
        self.assertEqual(" ".join(row.text for row in plan).strip(), "коротко")

    def test_tiny_tail_does_not_force_small_font(self):
        plan = fit_region(
            self.region([300, 300, 12]), "Первое предложение и второе предложение", Font
        )
        self.assertGreaterEqual(plan[0].font.size, 16)
        self.assertEqual(
            " ".join(row.text for row in plan).strip(),
            "Первое предложение и второе предложение",
        )
        self.assert_inside(plan)

    def test_small_ocr_height_does_not_shrink_other_rows(self):
        from dataclasses import replace

        region = self.region([300, 300, 300])
        region = replace(
            region, lines=(*region.lines[:2], replace(region.lines[2], bottom=53))
        )
        plan = fit_region(region, "Один два три четыре пять шесть", Font)
        self.assertGreaterEqual(plan[0].font.size, 16)
        self.assert_inside(plan)

    def test_impossible_word_rejects_whole_region(self):
        self.assertEqual(
            fit_region(self.region([10, 10]), "непомещающеесяслово текст", Font), []
        )

    def test_staggered_columns_do_not_merge(self):
        regions = group_ocr_texts(
            [OCRText("left", 0, 0, 100, 20), OCRText("right", 110, 24, 210, 44)]
        )
        self.assertEqual(len(regions), 2)

    def test_ragged_rows_and_last_line_use_natural_spacing(self):
        for widths in ([240, 180, 120], [240, 240, 120]):
            plan = fit_region(self.region(widths), "Один два три четыре пять шесть семь восемь", Font)
            self.assertTrue(plan)
            final = [row for row in plan if row.text][-1]
            self.assertEqual(len(final.runs), 1)
            if widths[1] == 180:
                self.assertTrue(all(len(row.runs) <= 1 for row in plan))
            self.assert_inside(plan)

    def test_strict_fit_measurements_scale_with_words_and_rows(self):
        from modules.text_processing.text_regions import split_translation_to_source_lines
        calls = []
        def measure(text):
            calls.append(text)
            return len(text) * 5
        text = " ".join(["слово"] * 100)
        result = split_translation_to_source_lines(self.region([300] * 12), text, measure, strict=True)
        self.assertEqual(" ".join(value for _, value in result).strip(), text)
        self.assertLessEqual(len(calls), 112)
