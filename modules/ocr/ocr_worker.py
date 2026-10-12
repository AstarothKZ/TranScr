from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable

import numpy as np

from modules.ocr.base import OCREngine
from modules.ocr.types import OCRText


class OCRWorker:
    """Выполняет OCR в фоне и не копит кадры в очереди."""

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
        self._job_lock = Lock()
        self._job_pending = False
        self._thread = Thread(target=self._run, name="ocr-worker", daemon=True)
        self._thread.start()

    def submit(self, image: np.ndarray, started_at: float) -> bool:
        """Передаёт кадр в OCR. Возвращает False, если очередь занята."""
        if self._stop_event.is_set():
            return False
        with self._job_lock:
            # Не принимаем второй кадр, пока не завершён предыдущий. Это также
            # не даёт результату старого кадра попасть в новый запрос приложения.
            if self._job_pending:
                return False
            self._job_pending = True
            try:
                self._queue.put_nowait((image, started_at))
                return True
            except Full:
                self._job_pending = False
                return False

    def close(self, timeout: float = 5.0) -> None:
        """Останавливает worker и ограничивает ожидание при зависшем OCR."""
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            print("Предупреждение: поток OCR ещё выполняется.", flush=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                image, started_at = self._queue.get(timeout=0.1)
            except Empty:
                continue

            try:
                texts = self._ocr.recognize(image)
                total_time_ms = (perf_counter() - started_at) * 1000
            except Exception as error:
                with self._job_lock:
                    self._job_pending = False
                if self._on_error is not None:
                    self._on_error(error)
                else:
                    print(f"Ошибка OCR: {error}", flush=True)
            else:
                with self._job_lock:
                    self._job_pending = False
                self._on_result(texts, total_time_ms)
