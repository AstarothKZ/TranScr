from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Callable

import numpy as np

from modules.ocr.base import OCREngine
from modules.ocr.types import OCRText


class OCRWorker:
    """Выполняет OCR в отдельном потоке, не блокируя основной цикл."""

    def __init__(
        self,
        ocr: OCREngine,
        on_result: Callable[[list[OCRText], float], None],
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._ocr = ocr
        self._on_result = on_result
        self._on_error = on_error
        self._queue: Queue[tuple[np.ndarray, float]] = Queue(maxsize=1)
        self._stop_event = Event()
        self._thread = Thread(
            target=self._run,
            name="ocr-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, image: np.ndarray, started_at: float) -> None:
        """Передаёт кадр в OCR. Пока worker занят, новые кадры не копятся."""
        try:
            self._queue.put_nowait((image, started_at))
        except Full:
            print("OCR занят, кадр пропущен.")

    def close(self) -> None:
        """Останавливает worker и ждёт завершения текущей задачи."""
        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                image, started_at = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                ocr_started = perf_counter()
                texts = self._ocr.recognize(image)
                ocr_time_ms = (perf_counter() - ocr_started) * 1000
                total_time_ms = (perf_counter() - started_at) * 1000
                self._on_result(texts, total_time_ms)
                print(f"Время OCR: {ocr_time_ms:.0f} мс")
            except Exception as error:
                if self._on_error is not None:
                    self._on_error(error)
