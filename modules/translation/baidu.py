"""Baidu Translate official API provider."""

import hashlib
import os
import secrets
import time

import httpx

from .text_chunks import split_text


class BaiduTranslateProvider:
    API_URL = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    MAX_CHARS_PER_REQUEST = 900
    MIN_REQUEST_INTERVAL = 1.05  # Standard Baidu API plan is limited to about 1 request/s.

    def __init__(self, from_code: str = "en", to_code: str = "ru", timeout: float = 15.0):
        self.from_code = from_code
        self.to_code = to_code
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout, follow_redirects=True)
        self._last_request_at = 0.0

    def translate(self, text: str) -> str:
        app_id = os.getenv("BAIDU_APP_ID", "").strip()
        api_key = os.getenv("BAIDU_API_KEY", "").strip()
        if not app_id or not api_key:
            raise RuntimeError(
                "Для Baidu задайте BAIDU_APP_ID и BAIDU_API_KEY в файле .env. "
                "Получите их в кабинете Baidu Translate Open Platform."
            )

        chunks = split_text(text, self.MAX_CHARS_PER_REQUEST)
        translations: list[str] = []
        for chunk in chunks:
            if not chunk.strip():
                translations.append(chunk)
                continue
            if not any(character.isalpha() for character in chunk):
                translations.append(chunk)
                continue
            translations.append(self._translate_chunk(chunk, app_id, api_key))
        return "\n".join(translations).strip()

    def _translate_chunk(self, text: str, app_id: str, api_key: str) -> str:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)

        salt = secrets.token_hex(8)
        signature_source = f"{app_id}{text}{salt}{api_key}"
        sign = hashlib.md5(signature_source.encode("utf-8")).hexdigest()
        payload = {
            "q": text,
            "from": self.from_code,
            "to": self.to_code,
            "appid": app_id,
            "salt": salt,
            "sign": sign,
        }
        try:
            response = self._client.post(self.API_URL, data=payload)
            self._last_request_at = time.monotonic()
            response.raise_for_status()
            data = response.json()
        except Exception as error:
            raise RuntimeError(f"Baidu Translate не сработал: {type(error).__name__}: {error}") from error

        if data.get("error_code"):
            code = data.get("error_code")
            message = data.get("error_msg", "неизвестная ошибка")
            raise RuntimeError(f"Baidu Translate: ошибка API {code}: {message}")

        items = data.get("trans_result", [])
        translated = "\n".join(item.get("dst", "") for item in items).strip()
        if not translated:
            raise RuntimeError("Baidu Translate вернул пустой результат.")
        return translated

    def close(self) -> None:
        self._client.close()
