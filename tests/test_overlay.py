import unittest
from unittest.mock import Mock, patch

from modules.ocr.types import OCRText
from modules.overlay.screen_overlay import ScreenOverlay
from modules.text_processing.text_regions import TextRegion


@patch('modules.overlay.screen_overlay.ImageTk.PhotoImage', Mock())
class OverlayTests(unittest.TestCase):
    @patch('modules.overlay.screen_overlay.render_row', side_effect=ValueError('clipped text'))
    def test_failed_text_render_never_paints_source_background(self, render):
        overlay = self.overlay()
        shown = overlay.show([(TextRegion('Original',0,0,200,24,16),'Перевод')], (0,0,300,100))
        self.assertEqual(shown, 0)
        overlay.canvas.create_rectangle.assert_not_called()
        overlay.canvas.create_image.assert_not_called()

    @patch('modules.overlay.screen_overlay.render_row')
    def test_whole_region_is_kept_if_one_row_fails(self, render):
        from PIL import Image
        render.side_effect = [Image.new('RGBA',(200,24),'black'), ValueError('missing text')]
        overlay = self.overlay()
        lines = (OCRText('First row',0,0,200,24),OCRText('Second row',0,28,200,52))
        region = TextRegion('First row\nSecond row',0,0,200,52,16,lines)
        overlay.show([(region,'Первое предложение и вторая строка текста')],(0,0,300,100))
        overlay.canvas.create_rectangle.assert_not_called()
        overlay.canvas.create_image.assert_not_called()

    def test_small_overlap_between_regions_does_not_drop_whole_block(self):
        overlay = self.overlay()
        first = OCRText("first", 0, 0, 100, 20)
        second = OCRText("second", 0, 18, 100, 38)
        overlay.show(
            [
                (TextRegion("first", 0, 0, 100, 20, 12, (first,)), "Первый"),
                (TextRegion("second", 0, 18, 100, 38, 12, (second,)), "Второй"),
            ],
            (1000, 600, 318, 102),
        )
        self.assertEqual(overlay.canvas.create_rectangle.call_count, 2)
        boxes = [
            call.args[:4] for call in overlay.canvas.create_rectangle.call_args_list
        ]
        self.assertLessEqual(boxes[0][3], boxes[1][1])

    def overlay(self):
        overlay = ScreenOverlay.__new__(ScreenOverlay)
        overlay._closed = False
        overlay.window = Mock()
        overlay.canvas = Mock()
        overlay._enable_click_through = Mock(return_value=True)
        overlay._set_native_window_bounds = Mock(return_value=True)
        return overlay

    def test_unfittable_text_does_not_crash(self):
        overlay = self.overlay()
        overlay.show(
            [(TextRegion("text", 0, 0, 20, 20, 12), "Оченьдлинноеслово")],
            (0, 0, 300, 200),
        )
        overlay.canvas.create_rectangle.assert_not_called()

    def test_overlapping_regions_do_not_crash(self):
        overlay = self.overlay()
        shown = overlay.show(
            [
                (TextRegion("first", 0, 0, 100, 20, 12), "Один"),
                (TextRegion("second", 0, 0, 100, 20, 12), "Два"),
            ],
            (0, 0, 300, 200),
        )
        self.assertEqual(shown, 1)
        self.assertEqual(overlay.canvas.create_rectangle.call_count, 1)


if __name__ == "__main__":
    unittest.main()
