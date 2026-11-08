from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OCRText:
    """Распознанная строка и геометрия для будущего overlay."""

    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float | None = None
    polygon: tuple[tuple[int, int], ...] | None = None
