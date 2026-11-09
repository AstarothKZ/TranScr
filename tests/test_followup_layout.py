import unittest
from modules.ocr.types import OCRText
from modules.text_processing.text_regions import group_ocr_texts
from modules.overlay.layout import fit_region
from modules.overlay.fonts import PixelFont
from modules.overlay.rendering import render_row
class FollowupTests(unittest.TestCase):
 def test_heading_description_separate(self):
  lines=[OCRText('Our games',30,203,155,230),OCRText('We hit the international big leagues',30,237,450,256)]
  self.assertEqual(len(group_ocr_texts(lines)),2)
 def test_cookie_policy_fits_without_disappearing(self):
  from modules.text_processing.text_regions import TextRegion
  region=TextRegion('Cookie Policy',0,0,70,18,12)
  plan=fit_region(region,'Политика использования файлов cookie',PixelFont)
  self.assertTrue(plan)
  self.assertEqual(render_row(plan[0],'white').size,(70,18))
