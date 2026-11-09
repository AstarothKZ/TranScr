import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from modules.overlay.styling import preserve_case, estimate_bold, strengthen_color
from modules.overlay.fonts import font_path, PixelFont
from modules.overlay.layout import fit_region
from modules.text_processing.text_regions import TextRegion


class StylingTests(unittest.TestCase):
    def test_casing_and_names(self):
        self.assertEqual(preserve_case('DEFENCE', 'защита'), 'ЗАЩИТА')
        self.assertEqual(preserve_case('cockpit space', 'Пространство В Кабине'), 'пространство в кабине')
        self.assertEqual(preserve_case('Sight Head Dimensions', 'размеры головки'), 'Размеры головки')
        self.assertEqual(preserve_case('November 5, 2026', '5 ноября 2026 г.'), '5 ноября 2026 г.')
        self.assertEqual(preserve_case('The Saab RGS system', 'система Saab RGS'), 'Система Saab RGS')

    def test_contrast_increases_for_light_and_dark_backgrounds(self):
        self.assertEqual(strengthen_color('#808080', '#ffffff'), '#767676')
        self.assertEqual(strengthen_color('#808080', '#000000'), '#8a8a8a')

    def test_bold_estimation_from_actual_pixels(self):
        text = 'Sample heading words'
        for bold in (False, True):
            font = ImageFont.truetype(str(font_path(bold)), 32)
            image = Image.new('RGB', (400, 50), 'white')
            ImageDraw.Draw(image).text((3, 3), text, font=font, fill='black', anchor='lt')
            region = TextRegion(text, 0, 0, 390, 44, 32)
            self.assertEqual(estimate_bold(np.array(image), region), bold)

    def test_bold_text_still_fits_actual_ink_height(self):
        region = TextRegion('TITLE', 0, 0, 240, 15, 12)
        row = fit_region(region, 'ЗАЩИТА ЙЁ gj', lambda size: PixelFont(size, bold=True))[0]
        bounds = row.font.bounds(row.text)
        self.assertLessEqual(row.runs[0][1] + bounds[3]-bounds[1], 15)
