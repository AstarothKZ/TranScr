import unittest
import numpy as np
from modules.ocr.labels import recover_labels
from modules.ocr.types import OCRText
from modules.text_processing.filters import should_translate
from modules.text_processing.text_regions import group_ocr_texts


class ContactLabelsTests(unittest.TestCase):
    def test_contact_captions_share_weight_but_keep_source_width(self):
        from modules.text_processing.geometry import prepare_lines
        image = np.full((220,300,3), 245, dtype=np.uint8)
        lines = []
        for y in (0,70,140):
            lines.extend([OCRText('Business', 10,y,80,y+20),
                          OCRText('biz@example.com',10,y+24,230,y+48)])
        captions = [l for l in prepare_lines(image,lines) if l.text=='Business']
        self.assertEqual([l.right for l in captions], [80]*3)
        self.assertTrue(all(l.bold is not None for l in captions))
        self.assertEqual(len({l.bold for l in captions}), 1)

    def test_addresses_remain_untouched_but_captions_translate(self):
        for text in ('biz@cdprojektred.com', 'support.cdprojektred.com',
                     'contact.community@cdprojektred.com',
                     'cdprojektred.com/product-safety', 'https://example.com/help'):
            self.assertFalse(should_translate(text), text)
        self.assertTrue(should_translate('Technical Support'))
        self.assertTrue(should_translate('Support'))
        self.assertTrue(should_translate('Visit example.com for help'))
        lines = [OCRText('Technical Support', 0, 0, 130, 20),
                 OCRText('support.example.com', 0, 24, 200, 46)]
        groups = group_ocr_texts([l for l in lines if should_translate(l.text)])
        self.assertEqual([g.text for g in groups], ['Technical Support'])

    def test_separated_footer_links_recover_in_separate_boxes(self):
        image = np.full((30, 400, 3), 20, dtype=np.uint8)
        for x in (5, 65, 125, 185, 245, 305):
            image[8:20, x:x+30] = 240
        calls = []
        def recognize(crop):
            calls.append(crop.shape)
            return [OCRText('Support', 2, 2, crop.shape[1]-2, crop.shape[0]-2)]
        result = recover_labels(image, [OCRText('merged footer links', 3, 5, 338, 23)], recognize)
        self.assertEqual(len(result), 6)
        for a,b in zip(result,result[1:]):
            self.assertLessEqual(a.right, b.left)
        self.assertTrue(all(type(l.left) is int for l in result))

    def test_body_paragraph_does_not_trigger_extra_ocr(self):
        image = np.full((30,400,3), 255, dtype=np.uint8)
        def forbidden(crop):
            self.fail('Extra OCR for light body paragraph')
        lines = [OCRText('A regular paragraph line', 0, 0, 400, 20)]
        self.assertEqual(recover_labels(image, lines, forbidden), lines)
