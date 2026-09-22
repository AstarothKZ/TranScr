from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Callable

import numpy as np

from modules.ocr.base import OCRModule
from modules.ocr.types import OCRText


class OCRWorker:
    """Выполняет OCR в отдельном потоке."""

    def __init__(
        self,
        ocr: OCRModule,
        on_result: Callable[[list[OCRText]], None],
    ) -> None:
        self._ocr = ocr
        self._on_result = on_result

        self._queue: Queue[tuple[np.ndarray, float]] = Queue(maxsize=1)
        self._stop_event = Event()

        self._thread = Thread(
            target=self._run,
            name="ocr-worker",
            daemon=True,
        )

        self._thread.start()

    def submit(
        self,
        image: np.ndarray,
        capture_started: float,
    ) -> None:
        """Отправляет кадр на OCR."""

        try:
            self._queue.put_nowait(
                (image, capture_started)
            )
        except Full:
            print("OCR занят, кадр пропущен.")

    def close(self) -> None:
        """Останавливает worker."""

        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                image, capture_started = self._queue.get(
                    timeout=0.1
                )
            except Empty:
                continue

            try:
                ocr_started = perf_counter()

                texts = self._ocr.recognize(image)

                ocr_time = (
                    perf_counter() - ocr_started
                ) * 1000

                total_time = (
                    perf_counter() - capture_started
                ) * 1000

                print(f"OCR: {ocr_time:.0f} мс")
                print(f"Всего: {total_time:.0f} мс")

                self._on_result(texts)

            except Exception as error:
                print(f"Ошибка OCR: {error}")