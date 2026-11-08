"""Google's keyless web translation endpoints, with a reserve endpoint.

These are web endpoints, not the paid Google Cloud Translation API. Their
availability and response format can change independently of this application.
"""

from threading import Lock

import httpx

from .text_chunks import split_text


class GoogleTranslateError(RuntimeError):
    """Neither Google endpoint returned a usable translation."""


class GoogleTranslateProvider:
    MAX_CHARS = 4500
    ENDPOINTS = (
        ("https://translate.googleapis.com/translate_a/single", "gtx"),
        ("https://clients5.google.com/translate_a/t", "dict-chrome-ex"),
    )

    def __init__(self, from_code="en", to_code="ru", timeout=15.0):
        self.from_code = from_code
        self.to_code = to_code
        self._preferred = 0
        # One in-flight Google operation, even when the facade schedules a
        # frame's regions concurrently. Reuse connections and the healthy URL.
        self._lock = Lock()
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )

    def translate(self, text):
        chunks = split_text(text, self.MAX_CHARS)
        if not chunks:
            return ""
        with self._lock:
            return "\n".join(self._translate_chunk(chunk) for chunk in chunks)

    def _translate_chunk(self, text):
        failures = []
        last_error = None
        for index in (self._preferred, 1-self._preferred):
            url, client = self.ENDPOINTS[index]
            params = {
                "client": client, "sl": self.from_code, "tl": self.to_code,
                "ie": "UTF-8", "oe": "UTF-8",
            }
            if index == 0:
                params["dt"] = "t"
            try:
                response = self._client.post(url, params=params, data={"q": text})
                response.raise_for_status()
                translated = self._parse(response.json())
            except (httpx.HTTPError, ValueError) as error:
                last_error = error
                detail = (f"HTTP {error.response.status_code}"
                          if isinstance(error, httpx.HTTPStatusError)
                          else type(error).__name__)
                failures.append(f"{index+1}: {detail}")
                continue
            self._preferred = index
            return translated
        raise GoogleTranslateError(
            "Google Translate недоступен (" + "; ".join(failures) + "). "
            "Проверьте соединение или выберите резервный переводчик."
        ) from last_error

    @staticmethod
    def _parse(payload):
        if not isinstance(payload, list) or not payload:
            raise ValueError("Google returned no translation")
        first = payload[0]
        if isinstance(first, str):
            translated = first
        elif isinstance(first, list):
            parts = []
            for segment in first:
                if isinstance(segment, str):
                    parts.append(segment)
                elif isinstance(segment, list) and segment and isinstance(segment[0], str):
                    parts.append(segment[0])
                else:
                    raise ValueError("Unexpected Google translation segment")
            translated = "".join(parts)
        else:
            raise ValueError("Unexpected Google response")
        if not translated.strip():
            raise ValueError("Google returned empty text")
        return translated.strip()

    def close(self):
        self._client.close()
