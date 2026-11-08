"""Pixel font metrics shared by the overlay and offline visual checks."""
from pathlib import Path
import os
from functools import lru_cache
from PIL import ImageFont


def font_path(bold=False):
    return Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / ('segoeuib.ttf' if bold else 'segoeui.ttf')


@lru_cache(maxsize=64)
def load_font(size, bold=False):
    """Reuse read-only FreeType faces across captures and style estimation."""
    return ImageFont.truetype(str(font_path(bold)), size)


class PixelFont:
    def __init__(self, size, bold=False):
        self.size = size
        self.bold = bold
        self.font = load_font(size, bold)
        # Use visible glyph extents, not the font's extra typographic leading.
        box = self.bounds('AgЙру')
        self.height = box[3] - box[1]

    def measure(self, text):
        box = self.bounds(text)
        return max(self.font.getlength(text), box[2]-box[0])

    def bounds(self, text):
        return self.font.getbbox(text, anchor='la')

    def metrics(self, name):
        return self.height
