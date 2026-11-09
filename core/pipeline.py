"""Screen capture → OCR → filtering → translation pipeline, independent of GUI."""

from __future__ import annotations

import hashlib
import logging
import numpy as np
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Empty, Queue
from time import perf_counter
from typing import TYPE_CHECKING

from modules.capture.types import ScreenBounds
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText
from modules.text_processing.geometry import prepare_lines
from modules.overlay.background import BackgroundStyle, prepare_background
from modules.text_processing.filters import (
    should_translate, translation_changed, untranslated_fragments,
)
from modules.text_processing.image_validation import has_source_ink
from modules.text_processing.text_regions import (
    TextRegion, group_ocr_texts, layout_slots, separate_row_overlaps,
)
from modules.translation.translator import Translator

LOGGER = logging.getLogger("transcr.pipeline")

if TYPE_CHECKING:
    from modules.capture.screen_capture import ScreenCapturer


@dataclass(slots=True)
class RuntimeBundle:
    settings: tuple[str, str, bool, str]
    ocr_worker: OCRWorker
    ocr_results: Queue
    translator: Translator
    argos_model_ready: bool = True


@dataclass(slots=True)
class CaptureResult:
    area: ScreenBounds
    translated_regions: list[tuple[TextRegion, str]]
    recognized_count: int
    filtered_count: int
    ocr_ms: float
    translate_ms: float
    capture_ms: float
    errors: list[str]
    screenshot_path: Path | None = None
    cache_hit: bool = False
    source_image: np.ndarray | None = None
    backgrounds: dict[tuple[int, int, int, int], BackgroundStyle] = field(default_factory=dict)
    background_ms: float = 0.0


class FrameCache:
    """Keep the last exact frame's OCR/translation result to make repeat captures cheap."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_key: tuple[ScreenBounds, bytes] | None = None
        self._last_result: CaptureResult | None = None

    @staticmethod
    def fingerprint(image) -> bytes:
        return hashlib.blake2s(memoryview(np.ascontiguousarray(image)), digest_size=16).digest()

    def get(
        self,
        area: ScreenBounds,
        fingerprint: bytes,
        capture_ms: float,
        screenshot_path: Path | None,
    ) -> CaptureResult | None:
        with self._lock:
            if self._last_key != (area, fingerprint) or self._last_result is None:
                return None
            return replace(
                self._last_result,
                capture_ms=capture_ms,
                ocr_ms=0.0,
                translate_ms=0.0,
                background_ms=0.0,
                screenshot_path=screenshot_path,
                cache_hit=True,
            )

    def put(
        self, area: ScreenBounds, fingerprint: bytes, result: CaptureResult
    ) -> None:
        with self._lock:
            self._last_key = (area, fingerprint)
            self._last_result = result

    def clear(self) -> None:
        with self._lock:
            self._last_key = None
            self._last_result = None


def process_capture(
    screen_capturer: ScreenCapturer,
    area: ScreenBounds,
    save_debug: bool,
    runtime: RuntimeBundle,
    frame_cache: FrameCache | None = None,
    cancel: threading.Event | None = None,
) -> CaptureResult:
    """Process one captured region, without making any Tkinter/UI calls."""
    capture_started = perf_counter()

    def cancelled_result() -> CaptureResult:
        return CaptureResult(area, [], 0, 0, 0.0, 0.0, 0.0, [])

    if cancel is not None and cancel.is_set():
        return cancelled_result()
    LOGGER.info("Захватываю экран: x=%s y=%s w=%s h=%s", *area)
    image, screenshot_path = screen_capturer.capture_region(area, save_debug=save_debug)
    capture_ms = (perf_counter() - capture_started) * 1000
    LOGGER.info("Кадр получен за %.0f мс", capture_ms)
    fingerprint = FrameCache.fingerprint(image) if frame_cache is not None else b""
    if frame_cache is not None:
        cached = frame_cache.get(area, fingerprint, capture_ms, screenshot_path)
        if cached is not None:
            LOGGER.info("Совпадение в кэше кадра: повторно использую OCR и перевод")
            return cached

    # A stale OCR callback should never be confused with this screenshot.
    while True:
        try:
            runtime.ocr_results.get_nowait()
        except Empty:
            break
    ocr_started = perf_counter()
    if not runtime.ocr_worker.submit(image, ocr_started):
        raise RuntimeError("OCR занят предыдущей задачей. Попробуйте ещё раз.")
    deadline = perf_counter() + 60
    while True:
        if cancel is not None and cancel.is_set():
            return cancelled_result()
        try:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                raise TimeoutError("OCR не ответил за 60 секунд.")
            kind, payload, elapsed_ms, token = runtime.ocr_results.get(
                timeout=min(remaining, 0.1)
            )
            if token == ocr_started:
                break
        except Empty:
            continue
    if kind == "error":
        raise RuntimeError(f"Ошибка OCR: {payload}")
    recognized: list[OCRText] = payload
    ocr_ms = float(elapsed_ms)
    LOGGER.info(
        "OCR завершён за %.0f мс; распознано строк: %d", ocr_ms, len(recognized)
    )
    if cancel is not None and cancel.is_set():
        return CaptureResult(
            area, [], len(recognized), 0, ocr_ms, 0.0, capture_ms, [], screenshot_path
        )
    if not recognized:
        result = CaptureResult(
            area, [], 0, 0, ocr_ms, 0.0, capture_ms, [], screenshot_path
        )
        if frame_cache is not None:
            frame_cache.put(area, fingerprint, result)
        return result

    # Filter individual OCR lines before grouping. Otherwise, a column of model
    # numbers/IDs next to real text could be merged into one translatable block.
    valid_lines = [item for item in recognized if has_source_ink(image, item)]
    if len(valid_lines) != len(recognized):
        LOGGER.info(
            "OCR: исключено рамок без исходного рисунка или за пределами кадра: %d",
            len(recognized) - len(valid_lines),
        )
    valid_lines = prepare_lines(image, valid_lines)
    meaningful_lines = [item for item in valid_lines if should_translate(item.text)]
    filtered_count = len(recognized) - len(meaningful_lines)
    regions = group_ocr_texts(meaningful_lines, context=valid_lines)
    eligible = [region for region in regions if should_translate(region.text)]
    LOGGER.info(
        "Фильтрация: исключено строк %d; сформировано текстовых областей %d",
        filtered_count,
        len(eligible),
    )
    if not eligible:
        result = CaptureResult(
            area,
            [],
            len(recognized),
            filtered_count,
            ocr_ms,
            0.0,
            capture_ms,
            [],
            screenshot_path,
        )
        if frame_cache is not None:
            frame_cache.put(area, fingerprint, result)
        return result

    # Translate repeated labels once, but preserve a separate screen position
    # for each occurrence. The facade has a thread-safe cache for later frames.
    unique_texts = list(dict.fromkeys(region.text.strip() for region in eligible))
    translate_started = perf_counter()
    LOGGER.info(
        "Перевожу %d уникальных областей (видимых позиций: %d)",
        len(unique_texts),
        len(eligible),
    )
    translated_by_text: dict[str, str] = {}
    errors: list[str] = []
    for source, translated in runtime.translator.translate_many(
        unique_texts, cancel
    ).items():
        if isinstance(translated, Exception):
            errors.append(f"{source[:48]!r}: {type(translated).__name__}: {translated}")
        elif translation_changed(source, translated):
            translated_by_text[source] = translated.strip()
    translate_ms = (perf_counter() - translate_started) * 1000
    LOGGER.info(
        "Перевод завершён за %.0f мс; изменённых результатов: %d; ошибок: %d",
        translate_ms,
        len(translated_by_text),
        len(errors),
    )

    translated_regions = [
        (region, translated_by_text[region.text.strip()])
        for region in eligible
        if region.text.strip() in translated_by_text
    ]
    if cancel is not None and cancel.is_set():
        return cancelled_result()
    background_started = perf_counter()
    backgrounds = {}
    lines = separate_row_overlaps([
        line for region, _ in translated_regions for line in layout_slots(region)
    ])
    for line in lines:
        if cancel is not None and cancel.is_set():
            return cancelled_result()
        box = (line.left, line.top, line.right, line.bottom)
        if box not in backgrounds:
            backgrounds[box] = prepare_background(image, box)
    background_ms = (perf_counter() - background_started) * 1000
    LOGGER.info("Восстановление фона: %.0f мс", background_ms)
    result = CaptureResult(
        area=area,
        translated_regions=translated_regions,
        recognized_count=len(recognized),
        filtered_count=filtered_count,
        ocr_ms=ocr_ms,
        translate_ms=translate_ms,
        capture_ms=capture_ms,
        errors=errors,
        source_image=image,
        screenshot_path=screenshot_path,
        backgrounds=backgrounds,
        background_ms=background_ms,
    )
    if (
        frame_cache is not None
        and (
            result.translated_regions
            or result.filtered_count == result.recognized_count
        )
        and not result.errors
        and not (cancel is not None and cancel.is_set())
        and not any(untranslated_fragments(text) for _, text in translated_regions)
    ):
        frame_cache.put(area, fingerprint, result)
    return result
