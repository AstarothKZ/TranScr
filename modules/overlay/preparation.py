"""Prepare one immutable-by-convention raster off the Tk/UI thread."""

from dataclasses import dataclass, replace
from time import perf_counter

from PIL import Image

from modules.text_processing.text_regions import layout_slots, separate_row_overlaps
from .background import prepare_background
from .fonts import PixelFont
from .layout import fit_region
from .rendering import render_row
from .styling import estimate_bold, preserve_case, strengthen_color


@dataclass(frozen=True, slots=True)
class DisplayStats:
    regions: int = 0
    rows: int = 0
    skipped: int = 0


@dataclass(slots=True)
class PreparedOverlay:
    image: Image.Image | None
    left: int
    top: int
    stats: DisplayStats
    background_ms: float = 0
    layout_ms: float = 0
    raster_ms: float = 0
    skipped_reasons: tuple[str, ...] = ()


def overlaps(first, second):
    return (
        first[0] < second[2]
        and second[0] < first[2]
        and first[1] < second[3]
        and second[1] < first[3]
    )


def prepare_overlay(
    translated_regions, size, source_image=None, *, backgrounds=None, cancel=None
):
    """Keep whole-block rejection and exact pixel-bound checks from the live renderer."""
    width, height = size
    if width <= 0 or height <= 0:
        raise ValueError(f"Некорректные размеры оверлея: {size}")

    def check():
        if cancel is not None and cancel.is_set():
            from modules.translation.batching import TranslationCancelled

            raise TranslationCancelled("Подготовка оверлея отменена")

    check()
    started = perf_counter()
    sources = [
        line for region, _ in translated_regions for line in layout_slots(region)
    ]
    corrected = iter(separate_row_overlaps(sources))
    regions = [
        (
            replace(region, lines=tuple(next(corrected) for _ in layout_slots(region))),
            text,
        )
        for region, text in translated_regions
    ]
    regions.sort(
        key=lambda item: (
            -sum(line.confidence or 0 for line in item[0].lines)
            / max(1, len(item[0].lines)),
            item[0].top,
            item[0].left,
        )
    )
    fonts, occupied, accepted, skipped = {}, [], [], []
    background_ms, raster_ms = 0.0, 0.0
    for region, text in regions:
        check()
        if not text.strip():
            skipped.append("empty")
            continue
        text = preserve_case(region.text, text)
        bold = estimate_bold(source_image, region)

        def make_font(size):
            key = (size, bold)
            if key not in fonts:
                fonts[key] = PixelFont(size, bold)
            return fonts[key]

        plan = fit_region(region, text, make_font)
        boxes = [
            (row.source.left, row.source.top, row.source.right, row.source.bottom)
            for row in plan
        ]
        if not plan or any(
            x1 < 0
            or y1 < 0
            or x2 > width
            or y2 > height
            or x1 < region.left
            or x2 > region.right
            or y1 < region.top
            or y2 > region.bottom
            for x1, y1, x2, y2 in boxes
        ):
            skipped.append("bounds_or_fit")
            continue
        if any(overlaps(box, used) for box in boxes for used in occupied):
            skipped.append("overlap")
            continue
        layers = []
        try:
            for row, box in zip(plan, boxes):
                check()
                before = perf_counter()
                style = backgrounds.get(box) if backgrounds is not None else None
                if style is None and source_image is not None:
                    style = prepare_background(source_image, box)
                background_ms += (perf_counter() - before) * 1000
                before = perf_counter()
                color = (
                    strengthen_color(style.text_color, style.color)
                    if style
                    else "#111111"
                )
                text_layer = render_row(row, color)
                if style is not None and style.patch is not None:
                    layer = style.patch.convert("RGBA")
                else:
                    layer = Image.new(
                        "RGBA", text_layer.size, style.color if style else "#ffffff"
                    )
                layer.alpha_composite(text_layer)
                layers.append((box, layer))
                raster_ms += (perf_counter() - before) * 1000
        except (ValueError, OSError):
            skipped.append("raster_bounds")
            continue
        accepted.extend(layers)
        occupied.extend(boxes)
    check()
    before = perf_counter()
    raster = None
    left = top = 0
    if accepted:
        left, top = (
            min(box[0] for box, _ in accepted),
            min(box[1] for box, _ in accepted),
        )
        right, bottom = (
            max(box[2] for box, _ in accepted),
            max(box[3] for box, _ in accepted),
        )
        raster = Image.new("RGBA", (right - left, bottom - top))
        for box, layer in accepted:
            raster.paste(layer, (box[0] - left, box[1] - top))
    raster_ms += (perf_counter() - before) * 1000
    total_ms = (perf_counter() - started) * 1000
    return PreparedOverlay(
        raster,
        left,
        top,
        DisplayStats(len(regions) - len(skipped), len(accepted), len(skipped)),
        background_ms,
        max(0, total_ms - background_ms - raster_ms),
        raster_ms,
        tuple(skipped),
    )
