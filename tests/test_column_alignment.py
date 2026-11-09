import unittest
from modules.ocr.types import OCRText
from modules.text_processing.text_regions import TextRegion, group_ocr_texts
from modules.overlay.layout import fit_region
from modules.overlay.fonts import PixelFont


class ColumnAlignmentTests(unittest.TestCase):
    def region(self, edges):
        lines=tuple(OCRText('source row',left,i*20,right,i*20+16)
                    for i,(left,right) in enumerate(edges))
        return TextRegion('source',min(l.left for l in lines),0,
                          max(l.right for l in lines),lines[-1].bottom,12,lines)

    def test_drop_cap_does_not_right_align_the_column(self):
        region=self.region([(90,370)]*3+[(40,370)]*9)
        self.assertEqual(region.alignment,'justify')

    def test_missing_words_do_not_change_alignment(self):
        region=self.region([(405,716)]+[(387,716)]*10+[(469,716)])
        self.assertEqual(region.alignment,'justify')

    def test_one_centered_ocr_outlier_does_not_center_column(self):
        region=self.region([(100,332)]+[(41,370)]*6)
        self.assertEqual(region.alignment,'justify')

    def test_actual_right_and_center_alignment_remain_available(self):
        self.assertEqual(self.region([(100,300),(150,300),(50,300)]).alignment,'right')
        self.assertEqual(self.region([(100,300),(150,250),(50,350)]).alignment,'center')

    def test_full_last_source_row_is_filled(self):
        region=self.region([(0,260)]*4)
        text=' '.join(['Проверка текста и расположения строк']*4)
        plan=fit_region(region,text,PixelFont)
        self.assertTrue(plan)
        for row in plan:
            self.assertGreater(len(row.runs),1)
            x,_,word=row.runs[-1]
            self.assertAlmostEqual(x+row.font.measure(word),row.source.right-1)

    def test_fragment_keeps_own_alignment_and_geometry(self):
        lines=[OCRText('full row',10,i*20,310,i*20+16) for i in range(6)]
        lines += [OCRText('fragment',60,150,310,166),
                  OCRText('next fragment',10,170,310,186),
                  OCRText('end fragment',10,190,310,206),
                  OCRText('other column',350,150,640,166)]
        regions=group_ocr_texts(lines)
        fragment=next(r for r in regions if r.text.startswith('fragment'))
        self.assertEqual(fragment.alignment,'right')
        self.assertEqual(fragment.lines[0].left, 60)
        self.assertTrue(all(l.right<=310 for l in fragment.lines))
