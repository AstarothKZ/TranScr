from pathlib import Path
from time import perf_counter

import keyboard

from modules.capture.screen_capture import ScreenCapturer
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText
from modules.translation.translation_worker import TranslationWorker
from modules.translation.types import TranslatedText


class Application:
    """Простой pipeline MVP: захват экрана -> OCR -> перевод -> консоль."""

    HOTKEY = "alt+t"
    EXIT_KEY = "esc"
    DEFAULT_OCR_LANGUAGE = "en-US"
    DEFAULT_SOURCE_LANGUAGE = "auto"
    DEFAULT_TARGET_LANGUAGE = "ru"

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        screenshot_dir = self.project_root / "test" / "screenshots"
        self.screen_capturer = ScreenCapturer(screenshot_dir)

        self.source_language = self.DEFAULT_SOURCE_LANGUAGE
        self.ocr = self._select_ocr()
        self.translator = self._select_translator()

        # Separate workers keep slow OCR and translation off the hotkey loop.
        self.ocr_worker = OCRWorker(
            self.ocr,
            self._on_ocr_result,
            self._on_ocr_error,
        )
        self.translation_worker = TranslationWorker(
            self.translator,
            self._on_translation_result,
            self._on_translation_error,
            source_lang=self.source_language,
            target_lang=self.DEFAULT_TARGET_LANGUAGE,
        )
        self.hotkey_handle = None

    def run(self) -> None:
        print("\nTranScr запущен.")
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
            self.translation_worker.close()
            print("TranScr остановлен.")

    def _select_ocr(self):
        print("\nВыберите OCR:")
        print("1 - Windows OCR (по умолчанию)")
        print("2 - PaddleOCR")
        while True:
            choice = input("Ваш выбор: ").strip() or "1"
            try:
                if choice == "1":
                    from modules.ocr.windows_ocr import WindowsOCRModule

                    engine = WindowsOCRModule(language=self.DEFAULT_OCR_LANGUAGE)
                    print("Выбран: Windows OCR")
                    return engine
                if choice == "2":
                    from modules.ocr.paddle_ocr import PaddleOCRModule

                    engine = PaddleOCRModule()
                    print("Выбран: PaddleOCR")
                    return engine
            except Exception as error:
                print(f"Не удалось запустить выбранный OCR: {error}")
                continue
            print("Ошибка: введите 1 или 2.")

    def _select_translator(self):
        print("\nВыберите переводчик:")
        print("1 - Google Translate (по умолчанию, интернет)")
        print("2 - Argos Translate (офлайн, требуется установленная языковая модель)")
        while True:
            choice = input("Ваш выбор: ").strip() or "1"
            try:
                if choice == "1":
                    from modules.translation.google import GoogleTranslator

                    translator = GoogleTranslator()
                    self.source_language = self.DEFAULT_SOURCE_LANGUAGE
                    print("Выбран: Google Translate")
                    return translator
                if choice == "2":
                    from modules.translation.argos import ArgosTranslator

                    source = input("Исходный язык Argos (например en): ").strip() or "en"
                    translator = ArgosTranslator()
                    # Argos needs an explicit source language and installed language pair.
                    translator.validate_language_pair(source, self.DEFAULT_TARGET_LANGUAGE)
                    self.source_language = source
                    print(
                        "Выбран: Argos Translate "
                        f"({source} -> {self.DEFAULT_TARGET_LANGUAGE})"
                    )
                    print("Проверьте, что локальная модель этой языковой пары установлена.")
                    return translator
            except Exception as error:
                print(f"Не удалось запустить выбранный переводчик: {error}")
                continue
            print("Ошибка: введите 1 или 2.")

    def _capture_screen(self) -> None:
        capture_started = perf_counter()
        try:
            image, path = self.screen_capturer.capture_full_screen()
            capture_time_ms = (perf_counter() - capture_started) * 1000
            print(
                f"Скриншот сохранён: {path} "
                f"(захват: {capture_time_ms:.0f} мс)"
            )
            self.ocr_worker.submit(image, capture_started)
        except Exception as error:
            print(f"Ошибка захвата: {error}")

    def _on_ocr_result(self, texts: list[OCRText], total_time_ms: float) -> None:
        print(f"OCR: {total_time_ms:.0f} мс от захвата до результата")
        if not texts:
            print("Текст не найден.")
            return

        print("Распознанный текст:")
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
        self.translation_worker.submit(texts)

    @staticmethod
    def _on_ocr_error(error: Exception) -> None:
        print(f"Ошибка OCR: {error}")

    @staticmethod
    def _on_translation_result(
        texts: list[TranslatedText],
        total_time_ms: float,
    ) -> None:
        print(f"Перевод: {total_time_ms:.0f} мс")
        if not texts:
            print("Перевод отсутствует.")
            return
        for item in texts:
            print(f"{item.source.text} -> {item.translated}")

    @staticmethod
    def _on_translation_error(error: Exception) -> None:
        print(f"Ошибка перевода: {error}")
