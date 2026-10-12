"""Group nearby OCR lines into larger, translatable screen regions."""

from dataclasses import dataclass
from statistics import median

from modules.ocr.types import OCRText


@dataclass(frozen=True, slots=True)
class TextRegion:
    """A group of OCR lines and its bounding rectangle in screenshot pixels."""

    text: str
    left: int
    top: int
    right: int
    bottom: int
    font_size: int


def group_ocr_texts(items: list[OCRText]) -> list[TextRegion]:
    """Join vertically adjacent lines while preserving their screen position.

    This intentionally uses a small set of geometric rules instead of complex
    layout detection. It is sufficient for the first overlay prototype and
    reduces the number of online translation requests for paragraphs.
    """
    ordered = sorted(
        (item for item in items if item.text.strip()),
        key=lambda item: (item.top, item.left),
    )
    if not ordered:
        return []

    groups: list[list[OCRText]] = []
    current = [ordered[0]]

    for item in ordered[1:]:
        previous = current[-1]
        previous_height = max(1, previous.bottom - previous.top)
        vertical_gap = item.top - previous.bottom
        horizontal_gap = max(
            0,
            previous.left - item.right,
            item.left - previous.right,
        )
        vertical_limit = max(7, min(14, round(previous_height * 0.8)))
        horizontal_limit = max(50, previous_height * 4)

        if vertical_gap <= vertical_limit and horizontal_gap <= horizontal_limit:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)

    regions: list[TextRegion] = []
    for group in groups:
        heights = [max(1, item.bottom - item.top) for item in group]
        regions.append(
            TextRegion(
                text="\n".join(item.text.strip() for item in group),
                left=min(item.left for item in group),
                top=min(item.top for item in group),
                right=max(item.right for item in group),
                bottom=max(item.bottom for item in group),
                # Use a line's height rather than the full paragraph height.
                font_size=max(10, min(28, round(median(heights) * 0.85))),
            )
        )
    return regions
