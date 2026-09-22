from pathlib import Path
from time import perf_counter

import keyboard

from modules.capture.screen_capture import ScreenCapturer
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText


class Application:
    """Главный цикл приложения."""

    HOTKEY = "alt+t"
    EXIT_KEY = "esc"

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]

        screenshot_dir = (
            self.project_root / "test" / "screenshots"
        )

        self.screen_capturer = ScreenCapturer(
            screenshot_dir
        )

        self.ocr = self._select_ocr()

        self.ocr_worker = OCRWorker(
            self.ocr,
            self._on_ocr_result,
        )

        self.hotkey_handle = None

    def run(self) -> None:
        print()
        print("TranScr запущен.")
        print(
            f"Нажмите {self.HOTKEY.upper()} "
            "для захвата экрана."
        )
        print(
            f"Нажмите {self.EXIT_KEY.upper()} "
            "для выхода."
        )

        self.hotkey_handle = keyboard.add_hotkey(
            self.HOTKEY,
            self._capture_screen,
        )

        try:
            keyboard.wait(self.EXIT_KEY)
        finally:
            if self.hotkey_handle is not None:
                keyboard.remove_hotkey(
                    self.hotkey_handle
                )

            self.ocr_worker.close()

        print("TranScr остановлен.")

    def _select_ocr(self):
        """Показывает меню выбора OCR."""

        print()
        print("Выберите OCR:")
        print("1 - Windows OCR")
        print("2 - PaddleOCR")

        while True:
            choice = input("Ваш выбор: ").strip()

            if choice == "1":
                from modules.ocr.windows_ocr import WindowsOCRModule

                print("Выбран: Windows OCR")

                return WindowsOCRModule(
                    language="en-US"
                )

            if choice == "2":
                from modules.ocr.paddle_ocr import PaddleOCRModule

                print("Выбран: PaddleOCR")

                return PaddleOCRModule()

            print("Ошибка: введите 1 или 2.")

    def _capture_screen(self) -> None:
        capture_started = perf_counter()

        try:
            image, path = (
                self.screen_capturer.capture_full_screen()
            )

            capture_time = (
                perf_counter() - capture_started
            ) * 1000

            print(
                f"Скриншот сохранён: {path} "
                f"(захват: {capture_time:.0f} мс)"
            )

            self.ocr_worker.submit(
                image,
                capture_started,
            )

        except Exception as error:
            print(f"Ошибка захвата: {error}")

    def _on_ocr_result(
        self,
        texts: list[OCRText],
    ) -> None:
        print()
        print("Распознанный текст:")

        if not texts:
            print("Текст не найден.")
            return

        for item in texts:
            print(
                f"[{item.left}, {item.top}, "
                f"{item.right}, {item.bottom}] "
                f"{item.text}"
            )