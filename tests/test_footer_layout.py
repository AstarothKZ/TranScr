import unittest
from unittest.mock import Mock
from PIL import Image, ImageDraw
from modules.ocr.types import OCRText
from modules.overlay.fonts import PixelFont
from modules.overlay.layout import fit_region
from modules.text_processing.text_regions import TextRegion
from modules.translation.translator import Translator


class FooterLayoutTests(unittest.TestCase):
    def test_navigation_labels_use_selected_provider(self):
        translator = Translator()
        self.addCleanup(translator.close)
        translator._mymemory_provider = Mock()
        labels = ['For investors','User Agreement','Privacy Policy','Cookie Policy','Store','Forum','Support']
        translator._mymemory_provider.translate.side_effect = lambda text: 'Ответ сервиса: '+text
        results = translator.translate_many(labels)
        self.assertEqual(results, {text:'Ответ сервиса: '+text for text in labels})
        self.assertEqual(translator._mymemory_provider.translate.call_count,len(labels))
        self.assertEqual(translator.translate_text('Store'), 'Ответ сервиса: Store')

    def test_footer_keeps_one_source_row_and_all_pixels(self):
        for text,translated,width in [('User Agreement','Условия использования',114),
                                      ('Privacy Policy','Политика приватности',99),
                                      ('Support','Поддержка',58)]:
            with self.subTest(text=text):
                source = OCRText(text,0,0,width,18)
                region = TextRegion(text,0,0,width,18,10,(source,))
                row = fit_region(region,translated,PixelFont)[0]
                self.assertEqual(row.source,source)
                self.assertEqual(len(row.runs),1)
                self.assertEqual(' '.join(value for _,_,value in row.runs),translated)
                image = Image.new('RGBA',(width+80,128))
                draw = ImageDraw.Draw(image)
                for x,y,value in row.runs:
                    draw.text((40+x,40+y-row.ink_top),value,font=row.font.font,fill='white',anchor='la')
                bounds=image.getbbox()
                self.assertGreaterEqual(bounds[0],40)
                self.assertGreaterEqual(bounds[1],40)
                self.assertLessEqual(bounds[2],40+width)
                self.assertLessEqual(bounds[3],58)
