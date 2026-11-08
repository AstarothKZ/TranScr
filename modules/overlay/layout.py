"""Measured line-bound layout, independent of Tk windows and native handles."""

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
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


def fit_region(
    region: TextRegion, text: str, make_font: Callable[[int], Any]
) -> list[RowPlan]:
    """Return a complete plan or nothing; never truncate translated content."""
    slots = layout_slots(region)
    if len(slots) == 1 and slots[0].label_size is not None:
        source = slots[0]
        # Preserve a readable common size; wrap navigation labels into their
        # verified empty area instead of shrinking long captions to 4 px.
        size = source.label_size
        while size >= 7:
            font = make_font(size)
            step = font.metrics('linespace')+2
            rows = max(1, (source.bottom-source.top)//step)
            virtual = tuple(replace(source, top=source.top+i*step,
                            bottom=source.top+(i+1)*step, label_size=None) for i in range(rows))
            wrapped = replace(region, lines=virtual)
            partition = split_translation_to_source_lines(wrapped,text,font.measure,strict=True)
            if partition:
                runs=[]
                for slot,value in partition:
                    if not value:
                        continue
                    box=font.bounds(value)
                    runs.append((source.left+1, slot.top+1-box[1], value))
                # Runs already refer to the font's actual draw origin.
                return [RowPlan(source,text,font,runs,0)]
            size -= 1
        return []
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
        return []
    # Balance once at the selected size, preserving the fast greedy probes.
    # This also keeps shorter translations spread over the source paragraph.
    balanced = split_translation_to_source_lines(region, text, font.measure)
    if balanced and all(font.measure(value) <= line.right-line.left-2 for line, value in balanced):
        partition = balanced
    result = []
    alignment = region.alignment
    last_text_row = max(i for i, (_, value) in enumerate(partition) if value)
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
        if (alignment == "justify" and index < last_text_row and len(words) > 1
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
