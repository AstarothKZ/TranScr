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
        self.font = load_font(size, bold)
        self._bounds = {}
        self._measurements = {}
        # Use visible glyph extents, not the font's extra typographic leading.
        box = self.bounds('AgЙру')
        self.height = box[3] - box[1]

    def measure(self, text):
        if text in self._measurements:
            return self._measurements[text]
        box = self.bounds(text)
        value = max(self.font.getlength(text), box[2]-box[0])
        if len(self._measurements) < 4096:
            self._measurements[text] = value
        return value

    def bounds(self, text):
        if text not in self._bounds:
            value = self.font.getbbox(text, anchor='la')
            if len(self._bounds) < 4096:
                self._bounds[text] = value
            return value
        return self._bounds[text]

    def metrics(self, name):
        return self.height
