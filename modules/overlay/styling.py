"""Conservative casing and initial visual font-weight estimation."""
import re
import numpy as np
from modules.overlay.fonts import load_font


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
    elif next((c for c in source if c.isalpha()), '').isupper():
        # Capitalize only a leading letter: an English month followed by a day
        # can become a Russian date starting with a digit, with a lower-case month.
        translated = re.sub(r'^(\s*)([^\W\d_])',
                            lambda m: m[1] + m[2].upper(), translated, count=1)
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
    # A region is a typographic flow. Five distributed rows suffice for its
    # weight vote; rendering both reference fonts for every row is expensive.
    sample = [lines[i] for i in sorted(set(np.linspace(0, len(lines)-1, min(5, len(lines)), dtype=int)))]
    for line in sample:
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
            font = load_font(max(8, line.bottom-line.top), bold)
            glyphs = np.array(font.getmask(line.text), dtype=np.uint8)
            refs.append(float((glyphs > 100).mean()) if glyphs.size else 0)
        if refs[1] > refs[0]:
            votes.append(density > refs[0] + (refs[1]-refs[0])*.65)
    return bool(votes) and bool(sum(votes) > len(votes)/2)
