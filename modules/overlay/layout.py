"""Measured line-bound layout, independent of Tk windows and native handles."""

from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from typing import Any

from modules.ocr.types import OCRText
from modules.text_processing.text_regions import (
    TextRegion,
    layout_slots,
    split_translation_to_source_lines,
)


@dataclass
class RowPlan:
    source: OCRText
    text: str
    font: Any
    # Text is always drawn using an explicit top-left anchor.
    runs: list[tuple[float, float, str]]
    ink_top: float = 0
    compressed: bool = False


def fit_region(
    region: TextRegion, text: str, make_font: Callable[[int], Any]
) -> list[RowPlan]:
    """Return a complete plan or nothing; never truncate translated content."""
    slots = layout_slots(region)
    if not slots or any(
        line.right <= line.left + 2 or line.bottom <= line.top for line in slots
    ):
        return []
    # Use the typical row height, so one noisy OCR box cannot shrink the block.
    # Narrow tail slots may be empty; their source background is still covered.
    min_height = int(median(line.bottom - line.top for line in slots))
    low, high = 5, max(5, round(min_height * 1.4))
    font, partition, selected_size = None, [], 0
    while low <= high:
        size = (low + high) // 2
        candidate = make_font(size)
        layout = (
            split_translation_to_source_lines(
                region, text, candidate.measure, strict=True
            )
            if candidate.metrics("linespace") <= min_height
            else []
        )
        if layout:
            font, partition, selected_size = candidate, layout, size
            low = size + 1
        else:
            high = size - 1
    if font is None:
        if len(slots) == 1 and slots[0].right-slots[0].left >= 24:
            source = slots[0]
            # A long single-line caption must not silently disappear. Render
            # its complete glyph run, then fit the raster inside its own box.
            return [RowPlan(source, text, make_font(max(5, min_height)), [], 0, True)]
        return []
    # Balance once at the selected size, preserving the fast greedy probes.
    # This also keeps shorter translations spread over the source paragraph.
    balanced = split_translation_to_source_lines(region, text, font.measure)
    if balanced and all(font.measure(value) <= line.right-line.left-2 for line, value in balanced):
        partition = balanced
    result = []
    alignment = region.alignment
    last_text_row = max(i for i, (_, value) in enumerate(partition) if value)
    typical_width = median(sorted(line.right-line.left for line in slots)[len(slots)//2:])
    for index, (source, row_text) in enumerate(partition):
        width = source.right - source.left - 2
        height = source.bottom - source.top
        row_font = font
        size = selected_size
        def ink_bounds():
            if hasattr(row_font, 'bounds'):
                return row_font.bounds(row_text)
            return (0, 0, row_font.measure(row_text), row_font.metrics('linespace'))

        box = ink_bounds()
        # Fit actual row glyphs, including accents and descenders. A sample
        # alphabet cannot guarantee the bounds of every translated string.
        padding = 1 if hasattr(row_font, 'bounds') and height >= 4 else 0
        while size > 1 and (box[3]-box[1] > height-2*padding
                            or row_font.measure(row_text) > width):
            size -= 1
            row_font = make_font(size)
            box = ink_bounds()
        if box[3]-box[1] > height or row_font.measure(row_text) > width:
            return []
        y = source.top + (height - (box[3]-box[1])) // 2
        x = source.left + 1 - min(0, box[0])
        words = row_text.split()
        runs = []
        # Justify only genuine body rows, never a final line or a ragged label.
        # Keep the source's occupied width for justified body rows.
        natural_space = row_font.measure(" ")
        sizes = [row_font.measure(word) for word in words]
        gap = (width - sum(sizes)) / max(1, len(words) - 1)
        fill_last = (region.alignment_hint == 'justify' or
                     (len(slots) >= 3 and source.right-source.left >= typical_width*.90
                      and slots[-1].right-slots[-1].left >= typical_width*.90))
        if (alignment == "justify" and (index < last_text_row or fill_last) and len(words) > 1
                and natural_space <= gap):
            for word, size in zip(words, sizes):
                runs.append((x, y, word))
                x += size + gap
        elif row_text:
            used = row_font.measure(row_text)
            if alignment == "right":
                x += width - used
            elif alignment == "center":
                x += (width - used) / 2
            runs.append((x, y, row_text))
        result.append(RowPlan(source, row_text, row_font, runs, box[1]))
    return result
