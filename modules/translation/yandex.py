"""Yandex Cloud Translate provider."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class YandexTranslateProvider:
    API_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"

    def __init__(self, from_code: str = "en", to_code: str = "ru", timeout: float = 15.0):
        self.from_code = from_code
        self.to_code = to_code
        self.timeout = timeout

    def translate(self, text: str) -> str:
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
            self.API_URL,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Yandex Cloud вернул HTTP {error.code}: {details}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(f"Не удалось подключиться к Yandex Cloud: {error}") from error

        translations = payload.get("translations", [])
        if not translations or not translations[0].get("text"):
            raise RuntimeError("Yandex Cloud вернул пустой результат перевода.")
        return translations[0]["text"]
