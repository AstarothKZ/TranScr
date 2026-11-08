"""Pixel font metrics shared by the overlay and offline visual checks."""
from pathlib import Path
import os
from PIL import ImageFont


def font_path(bold=False):
    return Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / ('segoeuib.ttf' if bold else 'segoeui.ttf')


class PixelFont:
    def __init__(self, size, bold=False):
        self.size = size
        self.bold = bold
        path = font_path(bold)
        self.font = ImageFont.truetype(str(path), size)
        # Use visible glyph extents, not the font's extra typographic leading.
        self.top = self.font.getbbox('AgЙру')[1]
        self.height = self.font.getbbox('AgЙру')[3] - self.top

    def measure(self, text):
        box = self.bounds(text)
        return max(self.font.getlength(text), box[2]-box[0])

    def bounds(self, text):
        return self.font.getbbox(text, anchor='la')

    def metrics(self, name):
        return self.height
