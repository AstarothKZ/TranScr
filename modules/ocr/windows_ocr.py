import asyncio

import numpy as np
from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import (
    BitmapPixelFormat,
    SoftwareBitmap,
)
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter

from modules.ocr.types import OCRText


class WindowsOCRModule:
    """OCR через встроенный Windows OCR."""

    def __init__(self, language: str = "en-US") -> None:
        language_object = Language(language)

        if not OcrEngine.is_language_supported(
            language_object
        ):
            available = [
                item.language_tag
                for item in OcrEngine.available_recognizer_languages
            ]

            raise RuntimeError(
                f"Язык OCR '{language}' не установлен.\n"
                f"Доступные языки: {available}"
            )

        self.engine = OcrEngine.try_create_from_language(
            language_object
        )

        if self.engine is None:
            raise RuntimeError(
                f"Не удалось создать Windows OCR "
                f"для языка '{language}'."
            )

    def recognize(
        self,
        image: np.ndarray,
    ) -> list[OCRText]:
        bitmap = self._create_bitmap(image)

        result = asyncio.run(
            self._recognize_async(bitmap)
        )

        texts: list[OCRText] = []

        for line in result.lines:
            words = [
                word
                for word in line.words
                if word.text.strip()
            ]

            if not words:
                continue

            left = min(
                int(word.bounding_rect.x)
                for word in words
            )

            top = min(
                int(word.bounding_rect.y)
                for word in words
            )

            right = max(
                int(
                    word.bounding_rect.x
                    + word.bounding_rect.width
                )
                for word in words
            )

            bottom = max(
                int(
                    word.bounding_rect.y
                    + word.bounding_rect.height
                )
                for word in words
            )

            text = line.text.strip()

            if not text:
                continue

            texts.append(
                OCRText(
                    text=text,
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    confidence=None,
                )
            )

        return texts

    async def _recognize_async(
        self,
        bitmap: SoftwareBitmap,
    ):
        return await self.engine.recognize_async(bitmap)

    @staticmethod
    def _create_bitmap(
        image: np.ndarray,
    ) -> SoftwareBitmap:
        height, width = image.shape[:2]

        rgba = np.empty(
            (height, width, 4),
            dtype=np.uint8,
        )

        rgba[:, :, :3] = image[:, :, :3]
        rgba[:, :, 3] = 255

        writer = DataWriter()
        writer.write_bytes(rgba.tobytes())

        buffer = writer.detach_buffer()

        return SoftwareBitmap.create_copy_from_buffer(
            buffer,
            BitmapPixelFormat.RGBA8,
            width,
            height,
        )