"""Explicit PNG export, independent of the OCR and desktop-capture pipeline."""

import re
from pathlib import Path

from PIL import Image

_OCR_NAMES = {"windows": "win", "paddle": "paddle"}
_PROVIDER_NAMES = {"google": "goog", "yandex": "yand", "argos": "argos"}


def screenshot_prefix(settings=None) -> str:
    if settings is None:
        return "empty"
    ocr, provider = settings[:2]
    return f"{_OCR_NAMES[ocr]}-{_PROVIDER_NAMES[provider]}"


def compose_screenshot(source, overlay=None) -> Image.Image:
    """Use the exact displayed raster, including its repaired source background."""
    frame = Image.fromarray(source).convert("RGBA")
    if overlay is not None and overlay.image is not None:
        frame.alpha_composite(overlay.image, (overlay.left, overlay.top))
    return frame.convert("RGB")


def save_screenshot(image: Image.Image, folder: Path, prefix: str) -> Path:
    """Number each OCR/provider series across restarts without overwriting files."""
    if not re.fullmatch(r"(?:empty|(?:win|paddle)-(?:goog|yand|argos))", prefix):
        raise ValueError("Некорректное имя серии снимков")
    folder.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(prefix) + r"-(\d+)\.png", re.IGNORECASE)
    number = (
        max(
            (
                int(match[1])
                for path in folder.iterdir()
                if (match := pattern.fullmatch(path.name))
            ),
            default=0,
        )
        + 1
    )
    while True:
        path = folder / f"{prefix}-{number}.png"
        try:
            output = path.open("xb")
        except FileExistsError:
            number += 1
            continue
        try:
            with output:
                image.save(output, format="PNG")
        except BaseException:
            path.unlink(missing_ok=True)
            raise
        return path
