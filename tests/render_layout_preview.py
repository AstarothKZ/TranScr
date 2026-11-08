"""Render a deterministic layout review with a real Windows font, without OCR/API."""

from pathlib import Path

from PIL import Image, ImageDraw

from modules.ocr.types import OCRText
from modules.overlay.layout import fit_region
from modules.overlay.fonts import PixelFont
from modules.overlay.rendering import render_row
from modules.text_processing.text_regions import TextRegion


def main():
    image = Image.new("RGB", (900, 430), "#eeeeee")
    draw = ImageDraw.Draw(image)
    text = "Перевод должен сохранять границы каждой строки и полностью закрывать исходный текст. Соседняя колонка остаётся отдельной областью."
    for offset, widths in [(20, [380, 380, 380, 220]), (470, [340, 300, 360, 190])]:
        lines = tuple(
            OCRText("original text", offset, 70 + i * 65, offset + w, 110 + i * 65)
            for i, w in enumerate(widths)
        )
        region = TextRegion(
            "original", offset, 70, offset + max(widths), 305, 26, lines
        )
        plan = fit_region(region, text, PixelFont)
        assert len(plan) == len(lines)
        for row in plan:
            line = row.source
            draw.rectangle(
                (line.left, line.top, line.right, line.bottom),
                fill="white",
                outline="#007c91",
            )
            layer = render_row(row, 'black')
            image.paste(layer, (line.left, line.top), layer)
    output = Path("test/line_layout_preview.png")
    output.parent.mkdir(exist_ok=True)
    image.save(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
