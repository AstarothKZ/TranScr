import unittest
import numpy as np
from modules.ocr.types import OCRText
from modules.text_processing.geometry import prepare_lines
from modules.text_processing.filters import should_translate


class GeometryTests(unittest.TestCase):
    def test_red_games_is_text_not_a_decorative_marker(self):
        image = np.full((45, 100, 3), 255, dtype=np.uint8)
        for x in range(10, 65, 10):
            image[10:30, x:x+6] = [235, 0, 10]
        line = prepare_lines(image, [OCRText('GAMES', 7, 7, 70, 33)])[0]
        self.assertLessEqual(line.left, 10)
        self.assertGreater(line.right-line.left, 50)
        self.assertTrue(should_translate(line.text))

    def test_png_is_removed_before_fragment_merging(self):
        image = np.full((40, 200, 3), 255, dtype=np.uint8)
        lines = prepare_lines(image, [OCRText('PNG', 0, 5, 25, 25),
                                     OCRText('205 KB', 30, 5, 100, 25)])
        self.assertFalse(any('PNG' in line.text for line in lines))
        self.assertTrue(all(not should_translate(line.text) for line in lines))

    def test_refinement_recovers_descender_without_next_row(self):
        image = np.full((60, 140, 3), 255, dtype=np.uint8)
        image[10:24, 20:25] = 30
        image[10:27, 40:45] = 30
        image[35:45, 20:25] = 30
        lines = prepare_lines(image, [OCRText('glyphs', 15, 8, 60, 24),
                                     OCRText('next row', 15, 33, 100, 47)])
        self.assertGreaterEqual(lines[0].bottom, 27)
        self.assertLess(lines[0].bottom, 33)

    def test_touching_heading_is_not_joined_to_body(self):
        from modules.text_processing.text_regions import group_ocr_texts
        lines = [OCRText('Sight Head dimensions', 0, 0, 190, 27),
                 OCRText('Length 259 mm width 136 mm', 0, 27, 350, 44)]
        self.assertEqual(len(group_ocr_texts(lines)), 2)

    def test_article_joins_same_baseline_before_short_filter(self):
        image = np.full((50, 300, 3), 255, dtype=np.uint8)
        lines = prepare_lines(image, [OCRText('The', 5, 5, 30, 25),
                                     OCRText('first aircraft', 36, 6, 200, 26)])
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0].text, 'The first aircraft')
        self.assertEqual(lines[0].left, 5)
        self.assertTrue(should_translate(lines[0].text))

    def test_colored_marker_is_outside_replacement_box(self):
        image = np.full((40, 200, 3), 255, dtype=np.uint8)
        image[10:20, 5:12] = [210, 60, 20]
        line = prepare_lines(image, [OCRText('》 Example text', 2, 5, 180, 25)])[0]
        self.assertEqual(line.text, 'Example text')
        self.assertGreater(line.left, 11)

    def test_short_logo_and_garbage_are_filtered(self):
        for value in ('cm', 'PC', 'HP', '河', 'A?', 'MP'):
            self.assertFalse(should_translate(value))
        self.assertTrue(should_translate('Open the PC settings'))

    def test_article_does_not_join_next_column(self):
        image = np.full((40, 300, 3), 255, dtype=np.uint8)
        lines = prepare_lines(image, [OCRText('The', 5, 5, 30, 25),
                                     OCRText('other column', 100, 5, 250, 25)])
        self.assertEqual(len(lines), 2)
