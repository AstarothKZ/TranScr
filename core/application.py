from pathlib import Path
from time import perf_counter

import keyboard

from modules.capture.screen_capture import ScreenCapturer
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText


class Application:
    """Главный цикл приложения MVP."""

    HOTKEY = "alt+t"
    EXIT_KEY = "esc"
    DEFAULT_OCR_LANGUAGE = "en-US"

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        screenshot_dir = self.project_root / "test" / "screenshots"

        self.screen_capturer = ScreenCapturer(screenshot_dir)
        self.ocr = self._select_ocr()
        self.ocr_worker = OCRWorker(
            self.ocr,
            on_result=self._on_ocr_result,
            on_error=self._on_ocr_error,
        )
        self.hotkey_handle = None

    def run(self) -> None:
        print()
        print("TranScr запущен.")
        print(f"{self.HOTKEY.upper()} — захват экрана.")
        print(f"{self.EXIT_KEY.upper()} — выход.")

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

    def _select_ocr(self):
        """Выбор OCR для текущего запуска."""
        print()
        print("Выберите OCR:")
        print("1 - Windows OCR (по умолчанию)")
        print("2 - PaddleOCR")

        while True:
            choice = input("Ваш выбор: ").strip() or "1"

            try:
                if choice == "1":
                    from modules.ocr.windows_ocr import WindowsOCRModule

                    ocr = WindowsOCRModule(language=self.DEFAULT_OCR_LANGUAGE)
                    print("Выбран: Windows OCR")
                    return ocr

                if choice == "2":
                    from modules.ocr.paddle_ocr import PaddleOCRModule

                    ocr = PaddleOCRModule()
                    print("Выбран: PaddleOCR")
                    return ocr
            except Exception as error:
                print(f"Не удалось запустить выбранный OCR: {error}")
                continue

            print("Ошибка: введите 1 или 2.")

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

    @staticmethod
    def _on_ocr_result(texts: list[OCRText], total_time_ms: float) -> None:
        print(f"OCR: {total_time_ms:.0f} мс")
        print("Распознанный текст:")

        if not texts:
            print("Текст не найден.")
            return

        for item in texts:
            confidence = (
                f" | conf={item.confidence:.3f}"
                if item.confidence is not None
                else ""
            )
            print(
                f"[{item.left}, {item.top}, {item.right}, {item.bottom}] "
                f"{item.text}{confidence}"
            )

    @staticmethod
    def _on_ocr_error(error: Exception) -> None:
        print(f"Ошибка OCR: {error}")
