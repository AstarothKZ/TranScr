"""Shared raster rendering for the real overlay and offline previews."""
from PIL import Image, ImageDraw


def render_row(row, color):
    line = row.source
    layer = Image.new('RGBA', (line.right-line.left, line.bottom-line.top))
    draw = ImageDraw.Draw(layer)
    for x, y, value in row.runs:
        draw.text((x-line.left, y-line.top-row.ink_top), value,
                  font=row.font.font, fill=color, anchor='la')
    return layer
