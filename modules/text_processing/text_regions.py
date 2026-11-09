"""GUI-independent OCR grouping and translation reflow in source coordinates.

The grouping deliberately prefers several small, correctly placed regions over
one large region that accidentally combines neighbouring columns.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from statistics import median

from modules.ocr.types import OCRText


@dataclass(frozen=True, slots=True)
class TextRegion:
    """A translatable region with the original geometry of every OCR line."""

    text: str
    left: int
    top: int
    right: int
    bottom: int
    font_size: int
    lines: tuple[OCRText, ...] = ()
    alignment_hint: str | None = None

    @property
    def width(self) -> int:
        return max(1, self.right - self.left)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.top)

    @property
    def looks_justified(self) -> bool:
        """Recognize stable column edges despite indents and incomplete OCR rows."""
        if len(self.lines) < 3:
            return False
        ordered = sorted(self.lines, key=lambda line: (line.top, line.left))
        heights = [max(1, line.bottom - line.top) for line in ordered]
        tolerance = max(3, round(median(heights) * 0.55))
        body = ordered[:-1]
        widths = sorted(line.right-line.left for line in body)
        reference = median(widths[len(widths)//2:])
        if reference < 20:
            return False
        # A few missed initial words must not turn a whole column into a
        # right-aligned paragraph. Look at the repeated full-width rows.
        full = [line for line in body if line.right-line.left >= reference*.90]
        if len(full) < max(1, len(body)*.60):
            return False
        left = median(line.left for line in full)
        right = median(line.right for line in full)
        stable = sum((abs(line.left-left) <= tolerance or
                      (line is ordered[0] and 0 <= line.left-left <= median(heights)*2))
                     and abs(line.right-right) <= tolerance for line in full)
        return stable >= len(full)*.75

    @property
    def alignment(self) -> str:
        """Estimate source alignment from repeated line geometry."""
        if self.alignment_hint is not None:
            return self.alignment_hint
        if self.looks_justified:
            return "justify"
        if len(self.lines) < 2:
            return "left"
        ordered = sorted(self.lines, key=lambda line: (line.top, line.left))
        heights = [max(1, line.bottom - line.top) for line in ordered]
        tolerance = max(3, round(median(heights) * 0.65))
        left_span = max(line.left for line in ordered) - min(line.left for line in ordered)
        right_span = max(line.right for line in ordered) - min(line.right for line in ordered)
        centers = [(line.left + line.right) / 2 for line in ordered]
        center_span = max(centers) - min(centers)
        if left_span <= tolerance:
            return "left"
        body_lefts = [line.left for line in ordered[1:]]
        if (
            len(ordered) >= 3
            and max(body_lefts) - min(body_lefts) <= tolerance
            and 0 <= ordered[0].left - min(body_lefts) <= median(heights) * 2
        ):
            return "left"
        if right_span <= tolerance:
            return "right"
        if center_span <= tolerance and left_span > tolerance*2 and right_span > tolerance*2:
            return "center"
        return "left"


def _valid_box(item: OCRText) -> bool:
    return bool(item.text.strip()) and item.right > item.left and item.bottom > item.top


def _pair_geometry(first: OCRText, second: OCRText) -> tuple[float, set[str]] | None:
    """Return vertical gap and plausible alignment modes for a consecutive row pair."""
    if first.top <= second.top:
        upper, lower = first, second
    else:
        upper, lower = second, first

    height_a = max(1, upper.bottom - upper.top)
    height_b = max(1, lower.bottom - lower.top)
    reference_height = float(median((height_a, height_b)))
    if max(height_a, height_b) / min(height_a, height_b) > 1.60:
        return None
    # A heading and its description are distinct translation units even when
    # the detector's padding makes their rectangles appear equally tall.
    if (upper.right-upper.left < (lower.right-lower.left)*.55
            and height_a > height_b*1.15):
        return None

    # Lines must be visually consecutive, not side-by-side items or paragraphs
    # separated by a blank line. Never group overlapping line boxes: their
    # rendered translations would have no guaranteed non-overlapping baseline.
    top_delta = lower.top - upper.top
    if top_delta < min(height_a, height_b) * 0.65:
        return None
    vertical_gap = lower.top - upper.bottom
    if vertical_gap < 0:
        return None
    if vertical_gap > max(3, reference_height * 0.80):
        return None

    x_tolerance = max(4, round(reference_height * 0.95))
    left_delta = abs(upper.left - lower.left)
    right_delta = abs(upper.right - lower.right)
    center_delta = abs((upper.left + upper.right) / 2 - (lower.left + lower.right) / 2)
    modes = set()
    if left_delta <= x_tolerance:
        modes.add("left")
    if right_delta <= x_tolerance:
        modes.add("right")
    if center_delta <= x_tolerance:
        modes.add("center")
    if not modes:
        return None

    # A large horizontal displacement usually means an adjacent column or a
    # callout. Require a stable alignment edge above in addition to this guard.
    horizontal_gap = max(0, upper.left - lower.right, lower.left - upper.right)
    if horizontal_gap > 0:
        return None
    return vertical_gap, modes


def _stable_group_alignment(lines: list[OCRText]) -> str | None:
    """Return a stable alignment edge for a flow, or None when geometry is ambiguous."""
    if len(lines) < 2:
        return None
    heights = [max(1, line.bottom - line.top) for line in lines]
    tolerance = max(4, round(median(heights) * 0.85))
    left_span = max(line.left for line in lines) - min(line.left for line in lines)
    right_span = max(line.right for line in lines) - min(line.right for line in lines)
    centers = [(line.left + line.right) / 2 for line in lines]
    center_span = max(centers) - min(centers)
    matches = []
    if left_span <= tolerance:
        matches.append((left_span, "left"))
    if right_span <= tolerance:
        matches.append((right_span, "right"))
    if center_span <= tolerance:
        matches.append((center_span, "center"))
    # Same-width lines can look both left- and right-aligned. Keep that pair
    # ambiguous until later lines make the dominant pattern clearer.
    if len(matches) != 1:
        return None
    return matches[0][1]


def _flow_score(first: OCRText, second: OCRText, preferred_mode: str | None) -> float:
    """Lower scores are better when a row can continue more than one text flow."""
    geometry = _pair_geometry(first, second)
    if geometry is None:
        return float("inf")
    vertical_gap, modes = geometry
    reference_height = max(
        1.0, median((first.bottom - first.top, second.bottom - second.top))
    )
    x_tolerance = max(4.0, reference_height * 0.95)
    if preferred_mode == "left":
        horizontal_delta = abs(first.left - second.left)
    elif preferred_mode == "right":
        horizontal_delta = abs(first.right - second.right)
    elif preferred_mode == "center":
        horizontal_delta = abs(
            (first.left + first.right) / 2 - (second.left + second.right) / 2
        )
    else:
        deltas = {
            "left": abs(first.left - second.left),
            "right": abs(first.right - second.right),
            "center": abs(
                (first.left + first.right) / 2 - (second.left + second.right) / 2
            ),
        }
        horizontal_delta = min(deltas[mode] for mode in modes)
    return horizontal_delta / x_tolerance + 0.2 * vertical_gap / reference_height


def separate_row_overlaps(items: list[OCRText]) -> list[OCRText]:
    """Split small vertical OCR-box overlaps without joining horizontal text."""
    result = list(items)
    ordered = sorted(range(len(result)), key=lambda index: result[index].top)
    for position, i in enumerate(ordered):
        for j in ordered[position+1:]:
            upper, lower = result[i], result[j]
            if items[j].top >= upper.bottom:
                break
            h = min(upper.bottom - upper.top, lower.bottom - lower.top)
            overlap = upper.bottom - lower.top
            shared = min(upper.right, lower.right) - max(upper.left, lower.left)
            if (
                h > 0
                and lower.top - upper.top >= h * 0.5
                and 0 < overlap <= h * 0.5
                and shared
                >= min(upper.right - upper.left, lower.right - lower.left) * 0.6
            ):
                boundary = (upper.bottom + lower.top) // 2
                result[i] = replace(upper, bottom=boundary)
                result[j] = replace(lower, top=boundary)
    return result


def group_ocr_texts(items: list[OCRText], *, context: list[OCRText] = ()) -> list[TextRegion]:
    """Group consecutive, similar-height rows into unambiguous vertical text flows.

    Unlike connected-components grouping, a row can join only the immediately
    preceding row of one existing flow. Ambiguous matches start a new region,
    favouring a few separate labels over merging adjacent columns into gibberish.
    """
    unique: dict[tuple[str, int, int, int, int], OCRText] = {}
    for item in separate_row_overlaps(items):
        if _valid_box(item):
            key = (item.text.strip(), item.left, item.top, item.right, item.bottom)
            unique.setdefault(key, item)
    ordered = sorted(
        unique.values(), key=lambda item: (item.top, item.left, item.bottom)
    )
    if not ordered:
        return []

    flows: list[list[OCRText]] = []
    # Numeric cells are filtered from translation but still describe the grid.
    # Keep neighbouring prose cells independent instead of reflowing a column
    # of statuses as a paragraph and moving translations between table rows.
    numeric_cells = [line for line in context
                     if line.text.strip() and all(c.isdigit() or c in ' .,/-' for c in line.text)]
    def table_cell(item):
        height = item.bottom-item.top
        return sum(min(item.bottom, cell.bottom)-max(item.top, cell.top)
                   >= .65*min(height, cell.bottom-cell.top)
                   and (cell.right <= item.left or cell.left >= item.right)
                   for cell in numeric_cells) >= 2

    table_cells = {id(item) for item in ordered if table_cell(item)}
    for item in ordered:
        if id(item) in table_cells:
            flows.append([item])
            continue
        candidates: list[tuple[float, int]] = []
        for flow_index, flow in enumerate(flows):
            previous = flow[-1]
            # Retired flows cannot resume after a blank line. Reject them
            # before computing medians and scanning their paragraph history.
            if item.top - previous.bottom > max(
                3, max(previous.bottom-previous.top, item.bottom-item.top) * .8
            ):
                continue
            if id(previous) in table_cells:
                continue
            # A short sentence-ending row followed by a full row starts a new
            # paragraph. Never redistribute its translation into the next one.
            body_width = max(row.right - row.left for row in flow)
            if (len(flow) >= 2
                    and item.left - min(row.left for row in flow) > (item.bottom-item.top)*.5
                    and previous.right-previous.left < body_width*.9):
                continue
            if (
                previous.text.rstrip().endswith((".", "!", "?", "。"))
                and previous.right - previous.left < body_width * 0.96
                and item.right - item.left > previous.right - previous.left + 4
            ):
                continue
            # A short, taller standalone heading above a full-width body row
            # is not the first line of that paragraph.
            if (
                len(flow) == 1
                and len(previous.text.split()) <= 6
                and not previous.text.rstrip().endswith((".", ",", ";", "-"))
                and item.right - item.left > body_width * 1.5
                and (previous.bottom-previous.top > (item.bottom-item.top)*1.15
                     or item.top-previous.bottom > min(previous.bottom-previous.top,
                                                       item.bottom-item.top)*.25)
            ):
                continue
            if len(flow) >= 2:
                gaps = [b.top - a.bottom for a, b in zip(flow, flow[1:])]
                typical_gap = median(gaps)
                if item.top - previous.bottom > typical_gap + max(
                    2, (previous.bottom - previous.top) * 0.25
                ):
                    continue
            geometry = _pair_geometry(previous, item)
            if geometry is None:
                continue
            modes = geometry[1]
            stable_mode = _stable_group_alignment(flow)
            if stable_mode is not None and stable_mode not in modes:
                continue
            candidates.append((_flow_score(previous, item, stable_mode), flow_index))

        if len(candidates) == 1:
            flows[candidates[0][1]].append(item)
        elif candidates:
            candidates.sort()
            # Only join a candidate if its visual alignment is clearly better
            # than the next flow. Otherwise the screen layout is ambiguous.
            best_score, best_index = candidates[0]
            second_score = candidates[1][0]
            if second_score - best_score >= 0.30:
                flows[best_index].append(item)
            else:
                flows.append([item])
        else:
            flows.append([item])

    regions: list[TextRegion] = []
    for component in flows:
        lines = tuple(sorted(component, key=lambda item: (item.top, item.left)))
        heights = [max(1, item.bottom - item.top) for item in lines]
        regions.append(
            TextRegion(
                text="\n".join(item.text.strip() for item in lines),
                left=min(item.left for item in lines),
                top=min(item.top for item in lines),
                right=max(item.right for item in lines),
                bottom=max(item.bottom for item in lines),
                font_size=max(5, min(28, round(median(heights) * 0.78))),
                lines=lines,
            )
        )
    return sorted(regions, key=lambda region: (region.top, region.left))


def layout_slots(region: TextRegion) -> list[OCRText]:
    """Keep every source rectangle intact; no horizontal expansion or merging."""
    return (
        sorted(region.lines, key=lambda line: (line.top, line.left))
        if region.lines
        else [
            OCRText(region.text, region.left, region.top, region.right, region.bottom)
        ]
    )


def split_translation_to_source_lines(
    region: TextRegion,
    translated_text: str,
    measure: Callable[[str], int],
    *,
    strict: bool = False,
) -> list[tuple[OCRText, str]]:
    """Partition all words over the original row widths without losing content.

    Strict fitting rejects overflow and permits empty slots for narrow tails;
    every source rectangle remains covered. Relaxed fitting penalizes overflow.
    """
    lines = layout_slots(region)
    words = translated_text.split()
    if not words:
        return []
    if len(lines) == 1:
        if strict and measure(" ".join(words)) > lines[0].right - lines[0].left - 2:
            return []
        return [(lines[0], " ".join(words))]
    count, rows = len(words), len(lines)
    widths = [max(1, line.right - line.left - 2) for line in lines]
    if strict:
        # Greedy packing is sufficient for feasibility: consuming more words in
        # an earlier slot cannot make the remaining suffix harder to fit.
        # Measure full runs (kerning included), without an O(rows * words²)
        # dynamic-programming search on every font-size probe.
        result = []
        start = 0
        for line, width in zip(lines, widths):
            end = start
            while end < count and measure(" ".join(words[start:end + 1])) <= width:
                end += 1
            result.append((line, " ".join(words[start:end])))
            start = end
        return result if start == count else []
    word_widths = [measure(word) for word in words]
    prefix = [0]
    for width in word_widths:
        prefix.append(prefix[-1] + width)
    space = measure(" ")
    dp = [{0: (0.0, -1)}]
    # Bounded search for large OCR paragraphs; each word remains in order.
    window = max(32, (count // rows + 1) * 4)
    for row, target in enumerate(widths):
        current = {}
        for start, (cost, _) in dp[-1].items():
            for end in range(start, min(count, start + window) + 1):
                n = end - start
                if n == 0 and count >= rows:
                    continue
                if count - end < rows - row - 1 and count >= rows:
                    continue
                actual = prefix[end] - prefix[start] + space * max(0, n - 1)
                ratio = actual / target
                penalty = (1 - ratio) ** 2 if ratio <= 1 else 8 * (ratio - 1) ** 2
                if n == 0:
                    penalty += 4
                value = cost + penalty
                if end not in current or value < current[end][0]:
                    current[end] = (value, start)
        dp.append(current)
    if count not in dp[-1]:
        return []
    result = []
    end = count
    for row in range(rows, 0, -1):
        start = dp[row][end][1]
        result.append((lines[row - 1], " ".join(words[start:end])))
        end = start
    return list(reversed(result))
