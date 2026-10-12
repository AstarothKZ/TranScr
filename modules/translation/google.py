"""Google Translate provider using the lightweight translate.googleapis.com endpoint.

The web endpoint is unofficial (like googletrans itself), but it avoids the
web-page token mechanism and lets httpx reuse its connection between captures.
"""

import re


class GoogleTranslateProvider:
    API_URL = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, from_code: str = "en", to_code: str = "ru", timeout: float = 15.0):
        try:
            import httpx
        except ImportError as error:
            raise RuntimeError(
                "Для Google Translate не установлен пакет httpx. "
                "Выполните: pip install -r requirements.txt"
            ) from error

        self.from_code = from_code
        self.to_code = to_code
        # Keep one HTTP client open. Reusing its connection pool makes later
        # captures faster than creating a new client for every translation.
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 TranScr/0.1"},
            follow_redirects=True,
        )

    def translate(self, text: str) -> str:
        try:
            response = self._client.get(
                self.API_URL,
                params={
                    "client": "gtx",
                    "sl": self.from_code,
                    "tl": self.to_code,
                    "dt": "t",
                    "q": text,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as error:
            raise RuntimeError(
                f"Google Translate не сработал: {type(error).__name__}: {error}"
            ) from error

        try:
            # The endpoint returns translation fragments; join them in order.
            fragments = payload[0]
            translated = "".join(
                fragment[0]
                for fragment in fragments
                if fragment and isinstance(fragment[0], str)
            ).strip()
        except (IndexError, TypeError, KeyError) as error:
            raise RuntimeError("Google Translate вернул ответ неизвестного формата.") from error

        if not translated:
            raise RuntimeError("Google Translate вернул пустой результат.")
        self._check_translation(text, translated)
        return translated

    def _check_translation(self, source: str, translated: str) -> None:
        """Don't report an identical English block as a successful translation."""
        if self.from_code == self.to_code:
            return
        normalized_source = " ".join(source.split()).casefold()
        normalized_result = " ".join(translated.split()).casefold()
        if normalized_source != normalized_result:
            return

        # Numeric lists, identifiers, and very short labels can legitimately
        # remain unchanged. Reject only word-rich blocks that should be translated.
        source_words = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", source)
        if len(source_words) >= 4:
            raise RuntimeError(
                "Google Translate вернул исходный текст без перевода. "
                "Проверьте интернет или повторите попытку; такой ответ не считается "
                "успешным переводом."
            )

    def close(self) -> None:
        """Release the reusable HTTP connection pool during application shutdown."""
        self._client.close()
