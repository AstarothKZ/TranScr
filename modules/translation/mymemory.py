"""MyMemory translation service using its HTTPS API directly."""

import os
from html import unescape

import httpx

from .text_chunks import split_text


class MyMemoryTranslateProvider:
    """Online translation without an API key; service quotas still apply."""

    API_URL = "https://api.mymemory.translated.net/get"
    MAX_BYTES_PER_REQUEST = 500  # UTF-8 bytes, not Unicode characters.

    def __init__(
        self, from_code: str = "en", to_code: str = "ru", timeout: float = 15.0
    ):
        self.from_code = from_code
        self.to_code = to_code
        self.email = os.getenv("MYMEMORY_EMAIL", "").strip()
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "TranScr/0.1"},
        )

    def translate(self, text: str) -> str:
        chunks = split_text(text, self.MAX_BYTES_PER_REQUEST, utf8_bytes=True)
        if not chunks:
            return ""

        # The facade owns concurrency; do not multiply pools per screen region.
        results = [
            self._translate_chunk(chunk) if any(c.isalpha() for c in chunk) else chunk
            for chunk in chunks
        ]

        translated = "\n".join(result or "" for result in results).strip()
        if not translated:
            raise RuntimeError("MyMemory вернул пустой перевод.")

        return translated

    def _translate_chunk(self, text: str) -> str:
        params = {
            "q": text,
            "langpair": f"{self.from_code}|{self.to_code}",
        }
        if self.email:
            # MyMemory accepts a contact email in `de` to raise the daily quota.
            params["de"] = self.email

        try:
            response = self._client.get(self.API_URL, params=params)
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise RuntimeError(
                f"MyMemory не сработал: {type(error).__name__}: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise RuntimeError("MyMemory вернул некорректный ответ.")
        status = payload.get("responseStatus", 200)
        details = payload.get("responseDetails", "")
        if str(status) != "200":
            raise RuntimeError(
                f"MyMemory вернул ошибку {status}: {details or 'без описания'}"
            )

        data = payload.get("responseData") or {}
        raw_text = data.get("translatedText") if isinstance(data, dict) else None
        if not isinstance(raw_text, str):
            raise RuntimeError("MyMemory вернул некорректный текст перевода.")
        translated = unescape(raw_text).strip()
        if not translated:
            raise RuntimeError(f"MyMemory вернул пустой результат. {details}".strip())
        if translated.upper().startswith("MYMEMORY WARNING"):
            raise RuntimeError(translated)
        return translated

    def close(self) -> None:
        self._client.close()
