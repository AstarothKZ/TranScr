"""Yandex Cloud Translate provider."""

import os

import httpx

from .text_chunks import split_text


class YandexTranslateProvider:
    API_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    MAX_CHARS = 10_000

    def __init__(
        self, from_code: str = "en", to_code: str = "ru", timeout: float = 15.0
    ):
        self.from_code = from_code
        self.to_code = to_code
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def translate(self, text: str) -> str:
        return self.translate_many([text])[0]

    def translate_many(self, texts: list[str]) -> list[str]:
        """Batch independent regions without merging their text or coordinates."""
        parts = [
            (index, chunk)
            for index, text in enumerate(texts)
            for chunk in split_text(text, self.MAX_CHARS)
        ]
        results: list[list[str]] = [[] for _ in texts]
        batch: list[tuple[int, str]] = []
        size = 0

        def flush() -> None:
            if batch:
                translated = self._request([part for _, part in batch])
                for (index, _), result in zip(batch, translated):
                    results[index].append(result)

        for item in parts:
            if size + len(item[1]) > self.MAX_CHARS:
                flush()
                batch = []
                size = 0
            batch.append(item)
            size += len(item[1])
        flush()
        return ["\n".join(result) for result in results]

    def _request(self, texts: list[str]) -> list[str]:
        api_key = os.getenv("YANDEX_API_KEY", "").strip()
        folder_id = os.getenv("YC_FOLDER_ID", "").strip()
        if not api_key or not folder_id:
            raise RuntimeError(
                "Для Яндекс-переводчика задайте YANDEX_API_KEY и YC_FOLDER_ID "
                "в файле .env. Пример находится в .env.example."
            )

        body = {
            "folderId": folder_id,
            "texts": texts,
            "sourceLanguageCode": self.from_code,
            "targetLanguageCode": self.to_code,
        }
        try:
            response = self._client.post(
                self.API_URL, json=body, headers={"Authorization": f"Api-Key {api_key}"}
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(
                f"Yandex Cloud вернул HTTP {error.response.status_code}"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise RuntimeError(
                f"Ошибка Yandex Cloud: {type(error).__name__}"
            ) from error
        translations = (
            payload.get("translations") if isinstance(payload, dict) else None
        )
        if not isinstance(translations, list) or len(translations) != len(texts):
            raise RuntimeError("Yandex Cloud вернул неверное количество переводов.")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("text"), str)
            or not item["text"].strip()
            for item in translations
        ):
            raise RuntimeError("Yandex Cloud вернул пустой или некорректный перевод.")
        return [item["text"] for item in translations]

    def close(self) -> None:
        self._client.close()
