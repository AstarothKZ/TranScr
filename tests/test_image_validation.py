import unittest

import numpy as np

from modules.ocr.types import OCRText
from modules.text_processing.image_validation import has_source_ink


class ImageValidationTests(unittest.TestCase):
    def test_blank_background_is_not_text(self):
        self.assertFalse(
            has_source_ink(
                np.full((20, 80, 3), 30, dtype=np.uint8), OCRText("ghost", 0, 0, 80, 20)
            )
        )

    def test_structure_at_offset_keeps_its_coordinates(self):
        image = np.zeros((100, 300, 3), dtype=np.uint8)
        image[50:60, 210:215] = 200
        self.assertTrue(has_source_ink(image, OCRText("text", 200, 45, 250, 70)))
        self.assertFalse(has_source_ink(image, OCRText("ghost", 0, 0, 50, 25)))

    def test_outside_frame_is_rejected(self):
        self.assertFalse(
            has_source_ink(
                np.zeros((20, 80, 3), dtype=np.uint8), OCRText("bad", -1, 0, 80, 20)
            )
        )
