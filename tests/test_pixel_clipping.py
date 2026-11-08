"""Verify real raster pixels rather than the font's sample metrics."""
import unittest
from PIL import Image, ImageDraw

from modules.overlay.fonts import PixelFont
from modules.overlay.layout import fit_region
from modules.overlay.rendering import render_row
from modules.text_processing.text_regions import TextRegion


class PixelClippingTests(unittest.TestCase):
    def test_actual_glyph_pixels_remain_inside_the_row(self):
        for text in ('защита', 'ЙЁ ÁÉ gjру', 'Недавно Royal заказали',
                     'А\u0301 е\u0308 g\u0323', 'ТРЕНАЖЁРЫ — обучение'):
            for height in (7, 9, 12, 16, 23):
                with self.subTest(text=text, height=height):
                    region = TextRegion('original', 0, 0, 400, height, 12)
                    plan = fit_region(region, text, PixelFont)
                    self.assertTrue(plan)
                    row = plan[0]
                    # Draw into a larger canvas: pixels escaping the row would
                    # otherwise silently disappear in the real overlay layer.
                    margin = 40
                    image = Image.new('RGBA', (480, height+80))
                    draw = ImageDraw.Draw(image)
                    for x, y, value in row.runs:
                        draw.text((margin+x, margin+y-row.ink_top), value,
                                  font=row.font.font, fill='black', anchor='la')
                    bounds = image.getbbox()
                    self.assertIsNotNone(bounds)
                    self.assertGreaterEqual(bounds[0], margin)
                    self.assertGreaterEqual(bounds[1], margin)
                    self.assertLessEqual(bounds[2], margin+400)
                    self.assertLessEqual(bounds[3], margin+height)
                    expected = image.crop((margin, margin, margin+400, margin+height))
                    self.assertEqual(render_row(row, 'black').tobytes(), expected.tobytes())
