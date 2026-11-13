"""Conservative casing and initial visual font-weight estimation."""
import re
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw
from modules.overlay.fonts import load_font


@lru_cache(maxsize=2048)
def glyph_density(text, size, bold):
    """Count glyph coverage in Pillow's C histogram, avoiding Python pixel copies."""
    font = load_font(size, bold)
    left, top, right, bottom = font.getbbox(text, anchor='la')
    width, height = right-left, bottom-top
    if width <= 0 or height <= 0:
        return 0.0
    image = Image.new('L', (width, height))
    ImageDraw.Draw(image).text((-left, -top), text, font=font, fill=255, anchor='la')
    return sum(image.histogram()[101:])/(width*height)


def strengthen_color(text_color, background_color):
    ink = np.array([int(text_color[i:i+2], 16) for i in (1, 3, 5)], dtype=float)
    bg = np.array([int(background_color[i:i+2], 16) for i in (1, 3, 5)], dtype=float)
    rgb = np.clip(np.rint(bg + (ink-bg)*1.08), 0, 255).astype(int)
    return '#%02x%02x%02x' % tuple(rgb)


def preserve_case(source, translated):
    letters = ''.join(c for c in source if c.isalpha())
    if not letters:
        return translated
    if letters.isupper():
        return translated.upper()
    if letters.islower():
        translated = translated.lower()
    elif source.istitle():
        translated = translated.title()
    elif next((c for c in source if c.isalpha()), '').isupper():
        translated = re.sub(r'[^\W\d_]', lambda m: m.group().upper(), translated, count=1)
    # Preserve the spelling of names/abbreviations present in both languages.
    originals = {word.casefold(): word for word in re.findall(r'[^\W\d_]+', source)}
    return re.sub(r'[^\W\d_]+', lambda m: originals.get(m.group().casefold(), m.group()), translated)


def estimate_bold(image, region):
    lines = getattr(region, 'lines', ()) or (region,)
    explicit = [line.bold for line in lines if getattr(line, 'bold', None) is not None]
    if explicit:
        return sum(explicit) >= len(explicit)/2
    if image is None:
        return False
    votes = []
    for line in lines:
        crop = image[line.top:line.bottom, line.left:line.right, :3]
        if not crop.size or line.bottom-line.top < 8:
            continue
        gray = crop.astype(float).mean(axis=2)
        low, high = np.percentile(gray, (5, 95))
        if high-low < 20:
            continue
        # Text normally occupies the smaller luminance cluster.
        dark = gray < (low+high)/2
        mask = dark if dark.mean() < .65 else ~dark
        yy, xx = np.nonzero(mask)
        if len(xx) < 8:
            continue
        density = mask[yy.min():yy.max()+1, xx.min():xx.max()+1].mean()
        refs = []
        for bold in (False, True):
            refs.append(glyph_density(line.text, max(8, line.bottom-line.top), bold))
        if refs[1] > refs[0]:
            votes.append(density > refs[0] + (refs[1]-refs[0])*.65)
    return bool(votes) and sum(votes) > len(votes)/2
