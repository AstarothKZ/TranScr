"""Yandex Cloud Translate provider."""

import os
from concurrent.futures import ThreadPoolExecutor

import httpx

from .batching import (
    RequestBudget,
    batch_results,
    make_client,
    map_bounded,
    pack_parts,
    text_parts,
    ServiceCooldown,
)


class YandexTranslateProvider:
    API_URL = "https://translate.api.cloud.yandex.net/translate/v2/translate"
    MAX_CHARS = 10_000

    def __init__(
        self, from_code: str = "en", to_code: str = "ru", timeout: float = 15.0
    ):
        self.from_code = from_code
        self.to_code = to_code
        self.timeout = timeout
        self._cooldown = ServiceCooldown("Yandex")
        self._client = make_client(timeout)
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="transcr-yandex"
        )

    def translate(self, text: str) -> str:
        result = self.translate_many([text])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def translate_many(
        self, texts: list[str], *, cancel=None, deadline=None
    ) -> list[str | Exception]:
        """Batch independent regions without merging their text or coordinates."""
        budget = RequestBudget(cancel, deadline)
        parts = text_parts(texts, self.MAX_CHARS)
        batches = pack_parts(parts, self.MAX_CHARS)

        def request_batch(batch):
            budget.check()
            self._cooldown.check()
            try:
                return self._request([part.text for part in batch], budget)
            except Exception as error:
                self._cooldown.record(error)
                raise

        values = map_bounded(
            self._executor,
            request_batch,
            batches,
            budget,
            2,
        )
        return batch_results(texts, parts, batches, values)

    def _request(
        self, texts: list[str], budget: RequestBudget | None = None
    ) -> list[str | Exception]:
        budget = budget or RequestBudget()
        budget.check()
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
                self.API_URL,
                json=body,
                headers={"Authorization": f"Api-Key {api_key}"},
                timeout=budget.timeout(self.timeout),
            )
            response.raise_for_status()
            payload = response.json()
            budget.check()
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
        return [
            item["text"].strip()
            if isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and item["text"].strip()
            else ValueError("Yandex Cloud вернул пустой или некорректный перевод")
            for item in translations
        ]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._client.close()
