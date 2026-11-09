"""Shared raster rendering for the real overlay and offline previews."""
from PIL import Image, ImageDraw


def render_row(row, color):
    line = row.source
    width, height = line.right-line.left, line.bottom-line.top
    if row.compressed:
        box = row.font.bounds(row.text)
        ink = Image.new('RGBA', (max(1, box[2]-box[0]), max(1, box[3]-box[1])))
        ImageDraw.Draw(ink).text((-box[0], -box[1]), row.text,
                                font=row.font.font, fill=color, anchor='la')
        target = (max(1,width-2), max(1,min(height-2, ink.height)))
        ink = ink.resize(target, Image.Resampling.LANCZOS)
        layer = Image.new('RGBA', (width,height))
        layer.paste(ink, (1,(height-target[1])//2))
        return layer
    margin = max(16, row.font.size*2)
    layer = Image.new('RGBA', (width+margin*2, height+margin*2))
    draw = ImageDraw.Draw(layer)
    for x, y, value in row.runs:
        draw.text((margin+x-line.left, margin+y-line.top-row.ink_top), value,
                  font=row.font.font, fill=color, anchor='la')
    bounds = layer.getbbox()
    if row.text.strip() and bounds is None:
        raise ValueError('Translation has no visible pixels')
    if bounds is not None and (bounds[0]<margin or bounds[1]<margin
                              or bounds[2]>margin+width or bounds[3]>margin+height):
        raise ValueError('Translation pixels extend outside the source row')
    return layer.crop((margin,margin,margin+width,margin+height))
