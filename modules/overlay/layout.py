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
    low, high = 5, min_height
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
    result = []
    for source, row_text in partition:
        width = source.right - source.left - 2
        height = source.bottom - source.top
        row_font = font
        size = selected_size
        while size > 1 and row_font.metrics("linespace") > height:
            size -= 1
            row_font = make_font(size)
        y = source.top + (height - row_font.metrics("linespace")) / 2
        x = source.left + 1
        words = row_text.split()
        runs = []
        # Preserve the occupied length of each source row. Single words cannot
        # be word-justified without changing their letter spacing.
        if len(slots) > 1 and len(words) > 1:
            sizes = [row_font.measure(word) for word in words]
            gap = (width - sum(sizes)) / (len(words) - 1)
            for word, size in zip(words, sizes):
                runs.append((x, y, word))
                x += size + gap
        elif row_text:
            used = row_font.measure(row_text)
            if region.alignment == "right":
                x += width - used
            elif region.alignment == "center":
                x += (width - used) / 2
            runs.append((x, y, row_text))
        result.append(RowPlan(source, row_text, row_font, runs))
    return result
