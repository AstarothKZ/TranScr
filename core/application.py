import threading
import time
import textwrap
from pathlib import Path
from time import perf_counter
import tkinter as tk

import keyboard

from modules.capture.screen_capture import ScreenCapturer
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText
from modules.translation.translator import Translator


class Application:
    """Связывает захват, OCR и перевод в простой последовательный сценарий."""

    HOTKEY = "alt+t"
    EXIT_KEY = "esc"
    DEFAULT_OCR_LANGUAGE = "en-US"

    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[1]
        screenshot_dir = self.project_root / "test" / "screenshots"

        self.root = tk.Tk()
        self.root.geometry("1x1+0+0")
        self.root.attributes("-alpha", 0.0)
        self.root.attributes("-topmost", True)
        self.root.withdraw()

        self.capture_requested = False
        self.should_exit = False
        self._ocr_done = threading.Event()
        self._ocr_texts: list[OCRText] = []
        self._ocr_error: Exception | None = None
        self._ocr_ms = 0.0

        self.ocr = self._select_ocr()
        self.ocr_worker = OCRWorker(
            self.ocr,
            on_result=self._on_ocr_result,
            on_error=self._on_ocr_error,
        )
        self.screen_capturer = ScreenCapturer(screenshot_dir)
        self.translator = Translator(
            root=self.root,
            from_code="en",
            to_code="ru",
            provider="google",
        )

        self.hotkey_handle = None
        self.exit_handle = None

    def run(self) -> None:
        self.translator.provider = self._select_translator()

        print()
        print("TranScr запущен.")
        print(f"Переводчик: {self.translator.PROVIDERS[self.translator.provider]}.")
        print(f"{self.HOTKEY.upper()} — захват экрана.")
        print(f"{self.EXIT_KEY.upper()} — выход.")

        self.hotkey_handle = keyboard.add_hotkey(self.HOTKEY, self.trigger_capture)
        self.exit_handle = keyboard.add_hotkey(self.EXIT_KEY, self.trigger_exit)

        try:
            while not self.should_exit:
                try:
                    self.root.update()
                except tk.TclError:
                    break

                if self.capture_requested:
                    self.capture_requested = False
                    self.translate()
                time.sleep(0.05)
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        for handle in (self.hotkey_handle, self.exit_handle):
            if handle is not None:
                try:
                    keyboard.remove_hotkey(handle)
                except Exception:
                    pass

        try:
            self.ocr_worker.close(timeout=5)
        except Exception as error:
            print(f"Предупреждение при остановке OCR: {error}", flush=True)

        try:
            self.root.destroy()
        except tk.TclError:
            pass
        print("TranScr остановлен.")

    def trigger_capture(self) -> None:
        self.capture_requested = True

    def trigger_exit(self) -> None:
        self.should_exit = True

    def capture(self) -> tuple[list[OCRText] | None, str | None]:
        try:
            image, path = self.screen_capturer.capture_full_screen()
            print(f"Скриншот сохранён: {path}", flush=True)

            self._ocr_done.clear()
            self._ocr_texts = []
            self._ocr_error = None
            self._ocr_ms = 0.0
            if not self.ocr_worker.submit(image, perf_counter()):
                print("OCR занят — захват пропущен.", flush=True)
                return None, None

            if not self._ocr_done.wait(timeout=60):
                print("OCR не ответил за 60 секунд.", flush=True)
                return None, None
            if self._ocr_error is not None:
                print(f"Распознавание OCR завершилось ошибкой: {self._ocr_error}", flush=True)
                return None, None

            texts = self._ocr_texts
            print(f"OCR: {self._ocr_ms:.0f} мс", flush=True)
            if not texts:
                print("Текст не найден.", flush=True)
                return None, None

            print("Распознанный текст:", flush=True)
            for item in texts:
                confidence = (
                    f" | conf={item.confidence:.3f}"
                    if item.confidence is not None else ""
                )
                print(
                    f"[{item.left}, {item.top}, {item.right}, {item.bottom}] "
                    f"{item.text}{confidence}",
                    flush=True,
                )

            # Сохраняем порядок строк, чтобы переводчик видел структуру текста.
            full_text = "\n".join(item.text for item in texts)
            print(f"Распознано {len(full_text)} символов.", flush=True)
            return texts, full_text
        except Exception as error:
            print(f"Ошибка захвата/OCR: {type(error).__name__}: {error}", flush=True)
            return None, None

    def translate(self) -> None:
        _, source_text = self.capture()
        if not source_text:
            return

        # Установка Argos нужна только при явном выборе офлайн-переводчика.
        if self.translator.provider == "argos" and not self.translator.check_language():
            try:
                self.root.deiconify()
                self.root.update()
                self.translator.manual_model_installation()
            except Exception as error:
                print(f"Не удалось открыть установку Argos: {error}", flush=True)
            finally:
                self.root.withdraw()

            if not self.translator.check_language():
                print("Модель Argos не установлена — перевод невозможен.", flush=True)
                return

        try:
            translated = self.translator.translate_text(source_text)
        except Exception as error:
            print(f"Ошибка перевода: {type(error).__name__}: {error}", flush=True)
            return

        print("Перевод на русский:", flush=True)
        print(textwrap.fill(translated, width=100), flush=True)

    def _on_ocr_result(self, texts: list[OCRText], total_time_ms: float) -> None:
        self._ocr_texts = texts
        self._ocr_ms = total_time_ms
        self._ocr_done.set()

    def _on_ocr_error(self, error: Exception) -> None:
        self._ocr_error = error
        self._ocr_done.set()

    def _select_ocr(self):
        print()
        print("Выберите OCR:")
        print("1 — Windows OCR (по умолчанию)")
        print("2 — PaddleOCR (дополнительная установка)")
        while True:
            choice = input("Ваш выбор [1]: ").strip() or "1"
            try:
                if choice == "1":
                    from modules.ocr.windows_ocr import WindowsOCRModule
                    ocr = WindowsOCRModule(language=self.DEFAULT_OCR_LANGUAGE)
                    print("Выбран Windows OCR.")
                    return ocr
                if choice == "2":
                    from modules.ocr.paddle_ocr import PaddleOCRModule
                    ocr = PaddleOCRModule()
                    print("Выбран PaddleOCR.")
                    return ocr
            except Exception as error:
                print(f"Не удалось запустить выбранный OCR: {error}")
                continue
            print("Введите 1 или 2.")

    def _select_translator(self) -> str:
        print()
        print("Выберите переводчик:")
        print("1 — Google Translate (по умолчанию, нужен интернет)")
        print("2 — Yandex Cloud Translate (нужны ключ API и ID каталога)")
        print("3 — Argos Translate (офлайн, нужен пакет и языковая модель)")
        print("4 — Гибридный режим (Google, затем Argos при сбое)")
        choices = {"1": "google", "2": "yandex", "3": "argos", "4": "hybrid"}
        while True:
            choice = input("Ваш выбор [1]: ").strip() or "1"
            if choice in choices:
                return choices[choice]
            print("Введите число от 1 до 4.")
