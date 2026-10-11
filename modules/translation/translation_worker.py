from queue import Empty, Full, Queue
from threading import Event, Thread
from time import perf_counter
from typing import Callable

from modules.ocr.types import OCRText
from modules.translation.base import Translator
from modules.translation.types import TranslatedText


class TranslationWorker:
    """Переводит OCR-результаты отдельно от OCR и горячих клавиш."""

    def __init__(
        self,
        translator: Translator,
        on_result: Callable[[list[TranslatedText], float], None],
        on_error: Callable[[Exception], None] | None = None,
        source_lang: str = "auto",
        target_lang: str = "ru",
    ) -> None:
        self._translator = translator
        self._on_result = on_result
        self._on_error = on_error
        self._source_lang = source_lang
        self._target_lang = target_lang
        self._queue: Queue[list[OCRText]] = Queue(maxsize=1)
        self._stop_event = Event()
        self._thread = Thread(
            target=self._run,
            name="translation-worker",
            daemon=True,
        )
        self._thread.start()

    def submit(self, texts: list[OCRText]) -> None:
        """Ставит OCR-текст в очередь; очередь ограничена одним результатом."""
        if self._stop_event.is_set():
            return
        try:
            self._queue.put_nowait(texts)
        except Full:
            print("Перевод занят, результат OCR пропущен.")

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                texts = self._queue.get(timeout=0.1)
            except Empty:
                continue

            started_at = perf_counter()
            translated: list[TranslatedText] = []
            for item in texts:
                try:
                    result = self._translator.translate(
                        item.text,
                        source_lang=self._source_lang,
                        target_lang=self._target_lang,
                    )
                    if result:
                        translated.append(TranslatedText(item, result))
                except Exception as error:
                    # A single bad line should not discard all other translations.
                    if self._on_error is not None:
                        self._on_error(error)
                    else:
                        print(f"Ошибка перевода строки '{item.text}': {error}")

            elapsed_ms = (perf_counter() - started_at) * 1000
            try:
                self._on_result(translated, elapsed_ms)
            except Exception as error:
                if self._on_error is not None:
                    self._on_error(error)
                else:
                    print(f"Ошибка обработки результата перевода: {error}")
