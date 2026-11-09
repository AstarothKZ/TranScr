import asyncio
import math
from dataclasses import replace
from PIL import Image
from typing import Any, cast

import numpy as np
from winsdk.windows.globalization import Language
from winsdk.windows.graphics.imaging import BitmapPixelFormat, SoftwareBitmap
from winsdk.windows.media.ocr import OcrEngine
from winsdk.windows.storage.streams import DataWriter

from modules.ocr.types import OCRText


class WindowsOCRModule:
    """OCR на основе встроенного компонента Windows Media OCR."""

    def __init__(self, language: str = "en-US") -> None:
        language_object = Language(language)
        if not OcrEngine.is_language_supported(language_object):
            available_languages = OcrEngine.available_recognizer_languages
            available = [item.language_tag for item in (available_languages or ())]
            raise RuntimeError(
                f"Язык OCR '{language}' не установлен. Доступные языки: {available}"
            )
        engine = OcrEngine.try_create_from_language(language_object)
        if engine is None:
            raise RuntimeError(
                f"Не удалось создать Windows OCR для языка '{language}'."
            )
        self.engine: OcrEngine = engine

    def recognize(self, image: np.ndarray, *, _split_labels=True) -> list[OCRText]:
        original = image
        scale = min(2.0, OcrEngine.max_image_dimension/max(image.shape[:2])) if _split_labels else 1.0
        if scale > 1:
            image = np.array(Image.fromarray(image).resize(
                (round(image.shape[1]*scale), round(image.shape[0]*scale)), Image.Resampling.LANCZOS))
        if max(image.shape[:2]) > OcrEngine.max_image_dimension:
            raise ValueError(
                f"Область превышает предел Windows OCR ({OcrEngine.max_image_dimension} px). "
                "Выберите меньшую область захвата."
            )
        bitmap = self._create_bitmap(image)
        try:
            result = asyncio.run(self._recognize_async(bitmap))
        finally:
            bitmap.close()
        texts: list[OCRText] = []

        for line in result.lines:
            words = [word for word in line.words if word.text.strip()]
            if not words:
                continue
            left = max(0, min(math.floor(word.bounding_rect.x) for word in words) - 1)
            top = max(0, min(math.floor(word.bounding_rect.y) for word in words) - 1)
            right = max(
                math.ceil(word.bounding_rect.x + word.bounding_rect.width) for word in words
            )
            bottom = max(
                math.ceil(word.bounding_rect.y + word.bounding_rect.height) for word in words
            )
            right = min(image.shape[1], right + 1)
            bottom = min(image.shape[0], bottom + 1)
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
                )
            )
        if _split_labels:
            if scale > 1:
                texts = [replace(item, left=max(0,math.floor(item.left/scale)),
                         top=max(0,math.floor(item.top/scale)),
                         right=min(original.shape[1],math.ceil(item.right/scale)),
                         bottom=min(original.shape[0],math.ceil(item.bottom/scale))) for item in texts]
            from modules.ocr.labels import recover_labels
            recognize = lambda crop: self.recognize(crop, _split_labels=False)
            return recover_labels(original, texts, recognize)
        return texts

    async def _recognize_async(self, bitmap: SoftwareBitmap) -> Any:
        # WinSDK's generated stub does not mark this WinRT operation as awaitable,
        # although the runtime object supports `await`.
        operation = cast(Any, self.engine).recognize_async(bitmap)
        return await operation

    @staticmethod
    def _create_bitmap(image: np.ndarray) -> SoftwareBitmap:
        height, width = image.shape[:2]
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = image[:, :, :3]
        rgba[:, :, 3] = 255

        writer = DataWriter()
        # The WinRT projection accepts a bytes buffer, but its generated type stub
        # incorrectly annotates this parameter as a list/array of integers.
        try:
            writer.write_bytes(cast(Any, rgba.tobytes()))
            buffer = writer.detach_buffer()
        finally:
            writer.close()
        bitmap = SoftwareBitmap.create_copy_from_buffer(
            buffer,
            BitmapPixelFormat.RGBA8,
            width,
            height,
        )
        if bitmap is None:
            raise RuntimeError("Windows не смог создать bitmap для OCR.")
        return bitmap
