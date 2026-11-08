"""Reject OCR detections whose source pixels contain no visible structure."""

import numpy as np


def has_source_ink(image, line) -> bool:
    height, width = image.shape[:2]
    if not (
        0 <= line.left < line.right <= width and 0 <= line.top < line.bottom <= height
    ):
        return False
    pixels = image[line.top : line.bottom, line.left : line.right, :3].reshape(-1, 3)
    # A flat background cannot contain text. Keep even low-contrast real text;
    # this check does not attempt to score recognition quality.
    return bool(np.max(np.ptp(pixels, axis=0)) >= 2)
