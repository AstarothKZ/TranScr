import unittest
import numpy as np
from PIL import Image, ImageDraw
from modules.overlay.background import prepare_background


class BackgroundTests(unittest.TestCase):
    def test_colored_solid_background_with_text(self):
        image = Image.new('RGB', (220, 40), (30, 70, 100))
        ImageDraw.Draw(image).text((8, 8), 'Original text', fill='white')
        style = prepare_background(np.array(image), (0, 0, 220, 40))
        self.assertIsNone(style.patch)
        self.assertEqual(style.color, '#1e4664')
        self.assertGreater(min(int(style.text_color[i:i+2], 16) for i in (1,3,5)), 230)

    def test_light_background_uses_dark_text(self):
        style = prepare_background(np.full((20, 100, 3), 240, dtype=np.uint8), (0, 0, 100, 20))
        self.assertIsNone(style.patch)
        self.assertEqual(style.text_color, '#111111')

    def test_texture_blurred_in_exact_box_without_mutating_frame(self):
        frame = np.random.default_rng(42).integers(30, 220, (60, 180, 3), dtype=np.uint8)
        before = frame.copy()
        style = prepare_background(frame, (10, 10, 170, 45))
        self.assertIsNotNone(style.patch)
        self.assertEqual(style.patch.size, (160, 35))
        self.assertLess(np.asarray(style.patch).std(), frame[10:45, 10:170].std())
        np.testing.assert_array_equal(frame, before)

    def test_patch_at_frame_edge(self):
        frame = np.random.default_rng(1).integers(0, 255, (20, 40, 3), dtype=np.uint8)
        self.assertEqual(prepare_background(frame, (0, 0, 40, 20)).patch.size, (40, 20))

    def test_bold_black_ink_does_not_become_background(self):
        frame = np.full((100, 240, 3), 255, dtype=np.uint8)
        frame[25:75, 30:210] = 20
        style = prepare_background(frame, (20, 20, 220, 80))
        self.assertIsNone(style.patch)
        self.assertEqual(style.color, '#ffffff')
        self.assertEqual(style.text_color, '#141414')

    def test_bold_ink_removed_from_gradient_without_dark_panel(self):
        gradient = np.linspace(155, 205, 240).astype(np.uint8)
        frame = np.broadcast_to(gradient[None, :, None], (100, 240, 3)).copy()
        frame[25:75, 30:210] = 20
        style = prepare_background(frame, (20, 20, 220, 80))
        self.assertIsNotNone(style.patch)
        self.assertGreater(np.asarray(style.patch).mean(), 150)
        self.assertEqual(style.text_color, '#141414')

    def test_dark_ink_on_gray_paper_keeps_dark_translation(self):
        frame = np.full((60, 200, 3), 120, dtype=np.uint8)
        frame[20:40, 20:180] = 25
        style = prepare_background(frame, (10, 15, 190, 45))
        self.assertEqual(style.text_color, '#191919')
