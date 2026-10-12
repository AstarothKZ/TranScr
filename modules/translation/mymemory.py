"""MyMemory translation service using its HTTPS API directly."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
import os
import re

import httpx

from .text_chunks import split_text


class MyMemoryTranslateProvider:
    """Online translation without an API key; service quotas still apply."""

    API_URL = "https://api.mymemory.translated.net/get"
    MAX_CHARS_PER_REQUEST = 450  # The service documents a 500-character query limit.
    MAX_WORKERS = 3

    def __init__(self, from_code: str = "en", to_code: str = "ru", timeout: float = 15.0):
        self.from_code = from_code
        self.to_code = to_code
        self.email = os.getenv("MYMEMORY_EMAIL", "").strip()
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "TranScr/0.1"},
        )

    def translate(self, text: str) -> str:
        chunks = split_text(text, self.MAX_CHARS_PER_REQUEST)
        if not chunks:
            return ""

        results = [None] * len(chunks)
        tasks = {}
        with ThreadPoolExecutor(max_workers=min(self.MAX_WORKERS, len(chunks))) as pool:
            for index, chunk in enumerate(chunks):
                # Numbers, IDs and punctuation do not need an external request.
                if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]", chunk):
                    results[index] = chunk
                else:
                    tasks[pool.submit(self._translate_chunk, chunk)] = index

            for future in as_completed(tasks):
                results[tasks[future]] = future.result()

        translated = "\n".join(result or "" for result in results).strip()
        if not translated:
            raise RuntimeError("MyMemory вернул пустой перевод.")

        source_norm = " ".join(text.split()).casefold()
        result_norm = " ".join(translated.split()).casefold()
        words = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", text)
        if source_norm == result_norm and len(words) >= 4:
            raise RuntimeError(
                "MyMemory вернул исходный текст без перевода. "
                "Попробуйте другой переводчик или проверьте доступность сервиса."
            )
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

        status = payload.get("responseStatus", 200)
        details = payload.get("responseDetails", "")
        if str(status) != "200":
            raise RuntimeError(f"MyMemory вернул ошибку {status}: {details or 'без описания'}")

        data = payload.get("responseData") or {}
        translated = unescape(str(data.get("translatedText", ""))).strip()
        if not translated:
            raise RuntimeError(f"MyMemory вернул пустой результат. {details}".strip())
        if translated.upper().startswith("MYMEMORY WARNING"):
            raise RuntimeError(translated)
        return translated

    def close(self) -> None:
        self._client.close()
