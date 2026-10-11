from dataclasses import dataclass

from modules.ocr.types import OCRText


@dataclass(frozen=True, slots=True)
class TranslatedText:
    """Перевод вместе с исходной строкой и её геометрией OCR."""

    source: OCRText
    translated: str
