import unittest
import numpy as np
from PIL import Image, ImageDraw
from modules.overlay.background import prepare_background


class BackgroundTests(unittest.TestCase):
    def test_tight_white_label_keeps_uniform_dark_exterior(self):
        from modules.overlay.fonts import PixelFont
        font = PixelFont(18, bold=True)
        image = Image.new('RGB', (100, 40), '#161616')
        draw = ImageDraw.Draw(image)
        draw.text((10, 8), 'Store', font=font.font, fill='white', anchor='lt')
        box = draw.textbbox((10, 8), 'Store', font=font.font, anchor='lt')
        box = (box[0]-1, box[1]-1, box[2]+1, box[3]+1)
        style = prepare_background(np.array(image), box)
        self.assertEqual(style.color, '#161616')
        self.assertGreater(int(style.text_color[1:3], 16), 200)

    def test_dense_white_letters_are_not_mistaken_for_paper(self):
        frame = np.full((40, 180, 3), 15, dtype=np.uint8)
        for x in range(10, 160, 20):
            frame[8:32, x:x+14] = 245
        style = prepare_background(frame, (5, 5, 170, 35))
        self.assertLess(int(style.color[1:3], 16), 40)
        self.assertGreater(int(style.text_color[1:3], 16), 220)

    def test_mixed_red_and_black_text_is_fully_erased(self):
        frame = np.full((40, 200, 3), 255, dtype=np.uint8)
        frame[12:28, 15:110] = [220, 10, 20]
        frame[12:28, 130:180] = 0
        style = prepare_background(frame, (5, 5, 195, 35))
        repaired = np.array(style.patch) if style.patch is not None else np.full((30,190,3), 255)
        self.assertGreater(repaired[10:20, 130:170].mean(), 230)

    def test_black_viewer_surround_does_not_override_gray_paper(self):
        frame = np.full((100, 260, 3), 25, dtype=np.uint8)
        frame[10:90, 10:250] = 155
        frame[25:75, 30:230] = 30
        style = prepare_background(frame, (10, 10, 250, 90))
        self.assertEqual(style.color, '#9b9b9b')
        self.assertEqual(style.text_color, '#1e1e1e')

    def test_adjacent_dark_rows_do_not_choose_white_text_on_paper(self):
        frame = np.full((36, 160, 3), 150, dtype=np.uint8)
        frame[6:10, 5:155] = 15
        frame[26:30, 5:155] = 15
        frame[15:21, 10:150] = 25
        style = prepare_background(frame, (5, 12, 155, 24))
        self.assertEqual(style.color, '#969696')
        self.assertEqual(style.text_color, '#191919')

    def test_dense_heading_repair_is_deterministic_and_keeps_paper_tone(self):
        rng = np.random.default_rng(12)
        paper = np.clip(160 + rng.normal(0, 4, (80, 220, 1)), 0, 255).astype(np.uint8)
        frame = np.repeat(paper, 3, axis=2)
        frame[20:60, 30:190] = 25
        before = frame.copy()
        first = np.asarray(prepare_background(frame, (10, 10, 210, 70)).patch)
        second = np.asarray(prepare_background(frame, (10, 10, 210, 70)).patch)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(frame, before)
        self.assertLess(abs(float(first.mean()) - 160), 6)
        self.assertGreater(first.std(), .5)

    def test_low_contrast_letters_on_textured_paper_are_erased(self):
        frame = np.full((80, 200, 3), 140, dtype=np.uint8)
        frame[:, ::3] = 146
        frame[25:55, 30:170] = 122
        style = prepare_background(frame, (20, 20, 180, 60))
        self.assertIsNotNone(style.patch)
        self.assertGreater(np.asarray(style.patch)[10:30, 20:140].mean(), 135)
        self.assertEqual(style.text_color, '#7a7a7a')

    def test_invalid_box_does_not_wrap_negative_coordinates(self):
        with self.assertRaises(ValueError):
            prepare_background(np.zeros((20, 30, 3), dtype=np.uint8), (-5, 0, 10, 10))

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
