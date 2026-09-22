from dataclasses import dataclass


@dataclass
class OCRText:
    text: str
    left: int
    top: int
    right: int
    bottom: int
    confidence: float | None = None