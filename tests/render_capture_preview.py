"""Offline visual QA: actual layout/background code, supplied OCR and translations.

No OCR models or network calls. Translation JSON maps a region's first OCR line
to a reviewed translation; omitted entries remain untouched in the source image.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from modules.ocr.types import OCRText
from modules.overlay.background import prepare_background
from modules.overlay.layout import fit_region
from modules.overlay.rendering import render_row
from modules.text_processing.text_regions import group_ocr_texts
from modules.overlay.fonts import PixelFont as Font
from modules.overlay.styling import preserve_case, estimate_bold, strengthen_color
from modules.text_processing.geometry import prepare_lines
from modules.text_processing.filters import should_translate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("ocr", type=Path)
    parser.add_argument("translations", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, help="Resize source to the OCR coordinate width")
    args = parser.parse_args()
    source = Image.open(args.image).convert("RGB")
    if args.width:
        source = source.resize((args.width, round(source.height * args.width / source.width)))
    data = json.loads(args.ocr.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        lines = [OCRText(text, max(0, box[0]-2), max(0, box[1]-2),
                         min(source.width, box[2]+2), min(source.height, box[3]+2))
                 for text, box in zip(data["rec_texts"], data["rec_boxes"])]
    else:
        lines = [OCRText(**item) for item in data]
    translations = json.loads(args.translations.read_text(encoding="utf-8"))
    frame = np.asarray(source)
    output = source.copy()
    plans = []
    started = perf_counter()
    prepared = prepare_lines(frame, lines)
    for region in group_ocr_texts([line for line in prepared if should_translate(line.text)], context=prepared):
        text = translations.get(region.text.splitlines()[0])
        if not text:
            continue
        bold = estimate_bold(frame, region)
        plan = fit_region(region, preserve_case(region.text, text), lambda size: Font(size, bold=bold))
        if not plan:
            raise ValueError(f"Translation does not fit: {region.text!r}")
        for row in plan:
            line = row.source
            box = (line.left, line.top, line.right, line.bottom)
            style = prepare_background(frame, box)
            output.paste(style.patch or Image.new("RGB", (box[2]-box[0], box[3]-box[1]), style.color), box)
            plans.append((row, strengthen_color(style.text_color, style.color)))
    for row, color in plans:
        layer = render_row(row, color)
        output.paste(layer, (row.source.left, row.source.top), layer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.save(args.output)
    print(f"{len(plans)} rows; {(perf_counter()-started)*1000:.1f} ms; {args.output.resolve()}")


if __name__ == "__main__":
    main()
