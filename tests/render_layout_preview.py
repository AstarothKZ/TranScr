"""Render a deterministic layout review with a real Windows font, without OCR/API."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from modules.ocr.types import OCRText
from modules.overlay.layout import fit_region
from modules.text_processing.text_regions import TextRegion


class Font:
    def __init__(self, size):
        self.font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", size)

    def measure(self, text):
        return self.font.getlength(text)

    def metrics(self, name):
        return sum(self.font.getmetrics())


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
        plan = fit_region(region, text, Font)
        assert len(plan) == len(lines)
        for row in plan:
            line = row.source
            draw.rectangle(
                (line.left, line.top, line.right, line.bottom),
                fill="white",
                outline="#007c91",
            )
            for x, y, value in row.runs:
                # Pillow's left-ascender anchor corresponds to the measured line box.
                draw.text((x, y), value, font=row.font.font, fill="black", anchor="lt")
    output = Path("test/line_layout_preview.png")
    output.parent.mkdir(exist_ok=True)
    image.save(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
