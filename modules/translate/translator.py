import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, filedialog
import socket
import argostranslate.package
import argostranslate.translate
import requests
import re
import argostranslate.sbd as sbd
from dotenv import load_dotenv

load_dotenv()
YANDEX_API_KEY = os.environ.get("YANDEX_API_KEY", "")
FOLDER_ID = os.environ.get("YC_FOLDER_ID", "")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def _split_sentences_regex(self,text):
    text = text.strip()
    if not text:
        return []
    return [p for p in re.split(r'(?<=[.!?])\s+', text) if p]


for _name in ['StanzaSentencizer', 'MiniSBDSentencizer', 'SpacySentencizerSmall', 'SBDetect']:
    _cls = getattr(sbd, _name, None)
    if _cls is not None and hasattr(_cls, 'split_sentences'):
        _cls.split_sentences = _split_sentences_regex



class Translator:

    def __init__(self, from_code, to_code, root=None):
        self.from_code = from_code
        self.to_code = to_code
        self.root = root
        self.api_key = YANDEX_API_KEY
        self.folder_id = FOLDER_ID
        self.models_dir = MODELS_DIR
        self.model_path = MODELS_DIR / f"translate-{from_code}_{to_code}.argosmodel"


    def ask_yes_no(self, prompt: str, title: str):
        return messagebox.askyesno(parent=self.root, title=title, message=prompt)


    def ask_file(self, initial_directory) -> str | None:
        path = filedialog.askopenfilename(
                parent=self.root,
                title="Выберите языковую модель (.argosmodel)",
                filetypes=[("Языковые модели Argos", "*.argosmodel")],
                initialdir=str(initial_directory),
        )
        return path or None

    def download_package_with_idle_timeout(self,package, connect_timeout=10, read_timeout=10):
        target_dir = Path(tempfile.gettempdir()) / "argos-translate"
        target_dir.mkdir(parents=True, exist_ok=True)
        for url in package.links:
            try:
                filename = url.split("/")[-1]
                print(filename)
                target_path = target_dir / filename
                total = 0
                last_report = 0

                with requests.get(url, stream=True, timeout=(connect_timeout, read_timeout)) as r:
                    r.raise_for_status()
                    total_size = int(r.headers.get("Content-Length", 0))
                    with open(target_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=64 * 1024):
                            if not chunk:
                                continue
                            f.write(chunk)
                            total += len(chunk)
                            mb = total / 1024 / 1024
                            if mb - last_report >= 10:
                                last_report = mb
                                if total_size:
                                    pct = total * 100 / total_size
                                    print(f"  ...{mb:.0f} МБ ({pct:.0f}%)", flush=True)
                                else:
                                    print(f"  ...{mb:.0f} МБ", flush=True)
                return str(target_path)
            except Exception as e:
                continue
        raise RuntimeError(f"Все зеркала недоступны")

    def manual_model_installation (self):
        if not self.ask_yes_no("Загрузить модель вручную?","Установка модели"):
            print("Установка отменена.", flush=True)
            return

        file_path = self.ask_file(Path.home() / "Downloads")
        if not file_path:
            print("Установка отменена.", flush=True)
            return
        try:
            argostranslate.package.install_from_path(file_path)
            print("Модель установлена.", flush=True)
        except Exception as err:
            print(f"Ошибка установки: {type(err).__name__}: {err}", flush=True)

    def get_translation(self):
        installed = argostranslate.translate.get_installed_languages()
        from_lang = [language for language in installed
                     if language.code == self.from_code][0]
        to_lang = [language for language in installed
                   if language.code == self.to_code][0]
        return from_lang.get_translation(to_lang)

    def translate_yandex_cloud(self, text: str):
        url = "https://translate.api.cloud.yandex.net/translate/v2/translate"
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        body = {
            "folderId": self.folder_id,
            "texts": [text],
            "sourceLanguageCode": self.from_code,
            "targetLanguageCode": self.to_code,
        }
        response = requests.post(url, headers=headers, json=body, timeout=10)
        response.raise_for_status()
        return response.json()["translations"][0]["text"]

    def download_and_get_available_packages(self):
        try:
            call_with_timeout(argostranslate.package.update_package_index, 30)
        except Exception as e:
            print(f"Не удалось обновить индекс: {type(e).__name__}: {e}", flush=True)
            return []
        return call_with_timeout(argostranslate.package.get_available_packages, 30)


    def download_model(self, available_packages):
        package = None
        for p in available_packages:
            if p.from_code == self.from_code and p.to_code == self.to_code:
                package = p
                break
        if package is None:
            raise LookupError(f"Пакет {self.from_code}→{self.to_code} не найден.")


        return self.download_package_with_idle_timeout(package)

    def install_model(self,downloaded_model_path):
        argostranslate.package.install_from_path(downloaded_model_path)


    def check_language(self):
        installed_languages = argostranslate.translate.get_installed_languages()
        for language in installed_languages:
            for translation in language.translations_from:
                if translation.from_lang.code == self.from_code and translation.to_lang.code == self.to_code:
                    return True
        return False



def call_with_timeout(func, timeout_sec):
    thread_pool_executor = ThreadPoolExecutor(max_workers=1)
    future = thread_pool_executor.submit(func)
    try:
        return future.result(timeout=timeout_sec)
    except TimeoutError:
        raise TimeoutError(f"Сервер не ответил за {timeout_sec} с.")
    finally:
        thread_pool_executor.shutdown(wait=False)


def is_online_service_available (host,timeout):
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(host)
        return True
    except (socket.gaierror, socket.timeout, OSError):
        return False
    finally:
        socket.setdefaulttimeout(None)


