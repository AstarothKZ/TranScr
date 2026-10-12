"""Простой переключаемый модуль перевода для TranScr.

Google Translate используется по умолчанию. Яндекс и Argos подключаются только
при выборе соответствующего режима, поэтому их дополнительные библиотеки не
мешают запуску базового прототипа.
"""

import asyncio
import json
import os
from pathlib import Path
from tkinter import filedialog, messagebox
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from dotenv import load_dotenv
except ImportError:  # .env необязателен; переменные среды тоже поддерживаются.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


class Translator:
    """Выбирает движок перевода, не связывая его с OCR или захватом экрана."""

    PROVIDERS = {
        "google": "Google Translate",
        "yandex": "Yandex Cloud Translate",
        "argos": "Argos Translate (офлайн)",
        "hybrid": "Гибридный режим (Google → Argos)",
    }

    YANDEX_API_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    def __init__(
        self,
        from_code: str = "en",
        to_code: str = "ru",
        root=None,
        provider: str = "google",
        timeout: float = 15.0,
    ) -> None:
        if provider not in self.PROVIDERS:
            raise ValueError(f"Неизвестный переводчик: {provider}")

        self.from_code = from_code
        self.to_code = to_code
        self.root = root
        self.provider = provider
        self.timeout = timeout

    def translate_text(self, text: str, provider: str | None = None) -> str:
        """Переводит текст выбранным движком и возвращает готовую строку."""
        text = text.strip()
        if not text:
            return ""

        selected = provider or self.provider
        if selected not in self.PROVIDERS:
            raise ValueError(f"Неизвестный переводчик: {selected}")

        if selected == "google":
            return self.translate_google(text)
        if selected == "yandex":
            return self.translate_yandex_cloud(text)
        if selected == "argos":
            return self.translate_argos(text)

        # Гибрид: сначала онлайн-перевод, при сбое — установленная офлайн-модель.
        try:
            return self.translate_google(text)
        except Exception as online_error:
            try:
                translated = self.translate_argos(text)
                print(
                    f"Google Translate недоступен; использован Argos. "
                    f"Причина: {type(online_error).__name__}",
                    flush=True,
                )
                return translated
            except Exception as offline_error:
                raise RuntimeError(
                    "Не удалось перевести текст через Google, а офлайн-перевод "
                    "Argos недоступен. Проверьте интернет и установку модели Argos. "
                    f"Google: {online_error}; Argos: {offline_error}"
                ) from offline_error

    def translate_google(self, text: str) -> str:
        """Переводит через библиотеку googletrans (интернет нужен)."""
        try:
            from googletrans import Translator as GoogleTranslator
        except ImportError as error:
            raise RuntimeError(
                "Google Translate не установлен. Выполните: "
                "pip install -r requirements.txt"
            ) from error

        async def translate_async() -> str:
            async with GoogleTranslator() as client:
                result = await client.translate(
                    text,
                    src=self.from_code,
                    dest=self.to_code,
                )
                return result.text

        try:
            return asyncio.run(translate_async())
        except Exception as error:
            raise RuntimeError(
                f"Google Translate не сработал: {type(error).__name__}: {error}"
            ) from error

    def translate_yandex_cloud(self, text: str) -> str:
        """Переводит через официальный REST API Yandex Cloud."""
        api_key = os.getenv("YANDEX_API_KEY", "").strip()
        folder_id = os.getenv("YC_FOLDER_ID", "").strip()
        if not api_key or not folder_id:
            raise RuntimeError(
                "Для Яндекс-переводчика задайте YANDEX_API_KEY и YC_FOLDER_ID "
                "в файле .env. Пример находится в .env.example."
            )

        body = {
            "folderId": folder_id,
            "texts": [text],
            "sourceLanguageCode": self.from_code,
            "targetLanguageCode": self.to_code,
        }
        request = Request(
            self.YANDEX_API_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Yandex Cloud вернул HTTP {error.code}: {details}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Не удалось подключиться к Yandex Cloud: {error}") from error

        translations = result.get("translations", [])
        if not translations or not translations[0].get("text"):
            raise RuntimeError("Yandex Cloud вернул пустой результат перевода.")
        return translations[0]["text"]

    def translate_argos(self, text: str) -> str:
        """Переводит локально с помощью установленной модели Argos Translate."""
        try:
            import argostranslate.translate as argos_translate
        except ImportError as error:
            raise RuntimeError(
                "Argos Translate не установлен. Для офлайн-перевода выполните: "
                "pip install -r requirements-optional.txt"
            ) from error

        installed_languages = argos_translate.get_installed_languages()
        source = next(
            (language for language in installed_languages if language.code == self.from_code),
            None,
        )
        if source is None:
            raise RuntimeError(f"В Argos не установлен язык: {self.from_code}")

        translation = next(
            (
                item
                for item in source.translations_from
                if item.to_lang.code == self.to_code
            ),
            None,
        )
        if translation is None:
            raise RuntimeError(
                f"В Argos не найдена модель {self.from_code} → {self.to_code}. "
                "Установите соответствующий языковой пакет."
            )
        return translation.translate(text)

    def check_language(self) -> bool:
        """Возвращает True, если модель Argos для текущей языковой пары установлена."""
        try:
            import argostranslate.translate as argos_translate
        except ImportError:
            return False

        for language in argos_translate.get_installed_languages():
            if language.code != self.from_code:
                continue
            return any(
                item.to_lang.code == self.to_code
                for item in language.translations_from
            )
        return False

    def manual_model_installation(self) -> bool:
        """Предлагает выбрать локальный файл модели Argos (.argosmodel)."""
        try:
            import argostranslate.package
        except ImportError:
            print(
                "Argos Translate не установлен. Выполните: "
                "pip install -r requirements-optional.txt",
                flush=True,
            )
            return False

        accepted = messagebox.askyesno(
            parent=self.root,
            title="Установка офлайн-переводчика",
            message="Выбрать файл языковой модели Argos (.argosmodel)?",
        )
        if not accepted:
            return False

        path = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите языковую модель Argos",
            filetypes=[("Языковые модели Argos", "*.argosmodel")],
            initialdir=str(Path.home() / "Downloads"),
        )
        if not path:
            return False

        try:
            argostranslate.package.install_from_path(path)
        except Exception as error:
            print(f"Не удалось установить модель Argos: {error}", flush=True)
            return False

        print("Модель Argos установлена.", flush=True)
        return self.check_language()
