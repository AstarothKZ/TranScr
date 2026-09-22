from pathlib import Path
from time import perf_counter

import keyboard

from modules.capture.screen_capture import ScreenCapturer
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.paddle_ocr import PaddleOCRModule


class Application:
    """Главный цикл приложения."""

    HOTKEY = "alt+t"
    EXIT_KEY = "esc"

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        screenshot_dir = self.project_root / "test" / "screenshots"

        self.screen_capturer = ScreenCapturer(screenshot_dir)

        # Инициализируем PaddleOCR в главном потоке.
        # Если окружение настроено неправильно, ошибка будет видна сразу.
        self.ocr = PaddleOCRModule()
        self.ocr_worker = OCRWorker(self.ocr)
        self.hotkey_handle = None

    def run(self) -> None:
        print("TranScr запущен.")
        print(f"Нажмите {self.HOTKEY.upper()} для захвата экрана.")
        print(f"Нажмите {self.EXIT_KEY.upper()} для выхода.")

        self.hotkey_handle = keyboard.add_hotkey(
            self.HOTKEY,
            self._capture_screen,
        )

        try:
            keyboard.wait(self.EXIT_KEY)
        finally:
            if self.hotkey_handle is not None:
                keyboard.remove_hotkey(self.hotkey_handle)

            self.ocr_worker.close()

        print("TranScr остановлен.")

    def _capture_screen(self) -> None:
        capture_started = perf_counter()

        try:
            image, path = self.screen_capturer.capture_full_screen()
            capture_time = (perf_counter() - capture_started) * 1000

            print(
                f"Скриншот сохранён: {path} "
                f"(захват: {capture_time:.0f} мс)"
            )

            self.ocr_worker.submit(image, capture_started)

        except Exception as error:
            print(f"Ошибка захвата: {error}")
