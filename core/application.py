import socket
import threading
import time
from pathlib import Path
from time import perf_counter
import tkinter as tk
import keyboard
from modules.capture.screen_capture import ScreenCapturer
from modules.ocr.ocr_worker import OCRWorker
from modules.ocr.types import OCRText
from modules.translate.translator import Translator

PING_HOSTS = [("77.88.8.8", 53), ("77.88.8.1", 53), ("1.1.1.1", 53), ("8.8.8.8", 53)]


def is_online_service_available (host,timeout):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False
    finally:
        socket.setdefaulttimeout(None)


def is_online(reason, timeout=4):
    for host, port in PING_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                if reason == "model_download":
                    ip = socket.gethostbyname("argos-net.com")
                    with socket.create_connection((ip, 443), timeout=timeout):
                        return True
                return True
        except (socket.timeout, OSError):
            continue
    return False

class Application:
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
        self.ocr = self._select_ocr()
        self.ocr_worker = OCRWorker(
            self.ocr,
            on_result=self._on_ocr_result,
            on_error=self._on_ocr_error,
        )

        self.screen_capturer = ScreenCapturer(screenshot_dir)
        self.translator = Translator(root=self.root, from_code="en", to_code="ru")
        self._ocr_done = threading.Event()
        self._ocr_texts: list[OCRText] = []
        self._ocr_ms: float = 0.0

        self.hotkey_handle = None
        self.exit_handle = None

    # ================= ЗАПУСК =================

    def run(self) -> None:
        is_offline_model_installed = self.translator.check_language()

        if not is_offline_model_installed and is_online("model_download",3):
            print ("Скачивание и установка офлайн-модели для возможности работы программы без интернета", flush=True)
            try:
                downloaded_model_path = self.translator.download_model(self.translator.download_and_get_available_packages())
                self.translator.install_model(downloaded_model_path)
            except Exception as e:
                print(f"{type(e).__name__}: {e}")
                print ("Не удалось скачать офлайн-модель автоматически. Рекомендуется сделать это вручную",flush=True)


        print()
        print("TranScr запущен.")
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
            self.ocr_worker.close()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        print("TranScr остановлен.")



    def trigger_capture(self) -> None:
        self.capture_requested = True

    def trigger_exit(self) -> None:
        self.should_exit = True

    def capture (self):
        try:
            image, path = self.screen_capturer.capture_full_screen()
            print(f"Скриншот сохранён: {path}", flush=True)
            self._ocr_done.clear()
            self._ocr_texts = []
            self._ocr_ms = 0.0
            self.ocr_worker.submit(image, perf_counter())

            if not self._ocr_done.wait(timeout=60):
                print("OCR не ответил за 60 сек.", flush=True)
                return None, None

            texts = self._ocr_texts
            total_time_ms = self._ocr_ms
            print(f"OCR: {total_time_ms:.0f} мс", flush=True)

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

            full_text = " ".join(item.text for item in texts)
            print(f"Распознано {len(full_text)} символов.", flush=True)
            return texts, full_text
        except Exception as error:
            print(f"Ошибка захвата/перевода: {error}", flush=True)
            return None, None


    def translate(self) -> None:
            ocr_blocks, one_string_text = self.capture()
            if one_string_text is None:
                return

            translated = None

            if is_online_service_available ("translate.api.cloud.yandex.net",3):
                try:
                    translated = self.translator.translate_yandex_cloud(one_string_text)
                except Exception as e:
                    print(f"Yandex Translate не сработал: {type(e).__name__}: {e}", flush=True)

            if translated is None:
                if not self.translator.check_language():
                    try:
                        self.root.deiconify()
                        self.root.update()
                        self.translator.manual_model_installation()
                    except Exception as error:
                        print(f"Ошибка при установке: {error}", flush=True)
                        return
                    finally:
                        self.root.withdraw()

                if not self.translator.check_language():
                    print("Модель не установлена — перевод невозможен.", flush=True)
                    return

                try:
                    translation_object = self.translator.get_translation()
                    translated = translation_object.translate(one_string_text)
                except Exception as e:
                    print(f"Ошибка перевода Argos: {e}", flush=True)
                    return

            print("RU:", flush=True)
            start = 0
            end = 0
            for letter in translated:
                end += 1
                if end - start > 100 and letter == ' ':
                    print(translated[start:end], flush=True)
                    start = end
            print(translated[start:], flush=True)

    def _on_ocr_result(self, texts: list[OCRText], total_time_ms: float) -> None:
        self._ocr_texts = texts
        self._ocr_ms = total_time_ms
        self._ocr_done.set()

    @staticmethod
    def _on_ocr_error(error: Exception) -> None:
        print(f"Ошибка OCR: {error}", flush=True)

    # ================= ВЫБОР OCR =================

    def _select_ocr(self):
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

