"""Keyless Google region batches, with validated mapping and reserve URLs."""

from threading import BoundedSemaphore, Lock
from time import perf_counter

import httpx

from .base import NetworkProvider
from .batching import (
    TranslationCancelled,
    join_parts,
    map_bounded,
    pack_parts,
    retry_delay,
    text_parts,
)


class GoogleTranslateError(RuntimeError):
    """No healthy Google endpoint returned a usable translation."""


class GoogleTranslateProvider(NetworkProvider):
    MAX_CHARS = 4000
    MAX_BATCH_ITEMS = 16
    ENDPOINTS = (
        ("https://translate.googleapis.com/translate_a/single", "gtx"),
        ("https://clients5.google.com/translate_a/t", "dict-chrome-ex"),
    )

    def __init__(self, from_code="en", to_code="ru", timeout=15.0):
        self.from_code, self.to_code = from_code, to_code
        self._state_lock = Lock()
        self._slots = BoundedSemaphore(2)
        self._cooldowns = {}
        self._batch_disabled_until = 0.0
        super().__init__(
            timeout,
            2,
            "transcr-google",
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )

    def translate_many(self, texts, *, cancel=None, deadline=None):
        budget = self._budget(cancel, deadline)
        parts = text_parts(texts, self.MAX_CHARS)
        batches = pack_parts(parts, self.MAX_CHARS, self.MAX_BATCH_ITEMS)
        values = map_bounded(
            self._executor,
            lambda batch: self._translate_batch(batch, budget),
            batches,
            budget,
            2,
        )
        # Failed batch requests fall back through the same bounded pool at
        # individual-part granularity. A slow tail cannot erase ready parts.
        translated, retry_parts, retry_positions = [], [], []
        for batch, value in zip(batches, values):
            if isinstance(value, Exception):
                if isinstance(value, (TranslationCancelled, TimeoutError)):
                    translated.extend([value] * len(batch))
                else:
                    retry_positions.extend(
                        range(len(translated), len(translated) + len(batch))
                    )
                    retry_parts.extend(batch)
                    translated.extend([value] * len(batch))
            else:
                for item, part in zip(value, batch):
                    if isinstance(item, Exception) and not isinstance(
                        item, (TranslationCancelled, TimeoutError)
                    ):
                        retry_positions.append(len(translated))
                        retry_parts.append(part)
                    translated.append(item)
        retried = map_bounded(
            self._executor,
            lambda part: self._translate_single(part.text, budget),
            retry_parts,
            budget,
            2,
        )
        for position, value in zip(retry_positions, retried):
            translated[position] = value
        return join_parts(texts, parts, translated)

    def _healthy(self, index):
        with self._state_lock:
            return perf_counter() >= self._cooldowns.get(index, 0)

    def _post(self, index, texts, budget):
        budget.check()
        if not self._healthy(index):
            raise GoogleTranslateError(
                "Google временно ограничил запросы к этому адресу"
            )
        while not self._slots.acquire(timeout=0.05):
            budget.check()
        try:
            budget.check()
            if not self._healthy(index):
                raise GoogleTranslateError(
                    "Google временно ограничил запросы к этому адресу"
                )
            url, client = self.ENDPOINTS[index]
            params = {
                "client": client,
                "sl": self.from_code,
                "tl": self.to_code,
                "ie": "UTF-8",
                "oe": "UTF-8",
            }
            if index == 0:
                params["dt"] = "t"
            return self._post_json(
                url,
                budget,
                params=params,
                data={"q": texts},
            )
        except (httpx.HTTPError, ValueError) as error:
            delay = 5.0
            if isinstance(error, httpx.HTTPStatusError):
                code = error.response.status_code
                delay = 30.0 if code in {403, 429} else 5.0
                delay = retry_delay(error.response.headers.get("Retry-After"), delay)
            with self._state_lock:
                self._cooldowns[index] = perf_counter() + delay
            raise
        finally:
            self._slots.release()

    def _translate_batch(self, batch, budget):
        budget.check()
        with self._state_lock:
            batch_enabled = perf_counter() >= self._batch_disabled_until
        if not batch_enabled or not self._healthy(1):
            raise GoogleTranslateError("Пакетный адрес Google временно недоступен")
        try:
            payload = self._post(1, [part.text for part in batch], budget)
            return self._parse_batch(payload, len(batch))
        except ValueError:
            with self._state_lock:
                self._batch_disabled_until = perf_counter() + 60
            raise

    def _translate_single(self, text, budget):
        failures, last_error = [], None
        for index in (0, 1):
            budget.check()
            if not self._healthy(index):
                continue
            try:
                payload = self._post(index, text, budget)
                value = (
                    self._parse(payload)
                    if index == 0
                    else self._parse_batch(payload, 1)[0]
                )
                if isinstance(value, Exception):
                    raise value
                return value
            except (TranslationCancelled, TimeoutError):
                raise
            except (httpx.HTTPError, ValueError, GoogleTranslateError) as error:
                last_error = error
                detail = (
                    f"HTTP {error.response.status_code}"
                    if isinstance(error, httpx.HTTPStatusError)
                    else type(error).__name__
                )
                failures.append(f"{index + 1}: {detail}")
        raise GoogleTranslateError(
            "Google Translate недоступен ("
            + ", ".join(failures or ["пауза после ошибки"])
            + ")"
        ) from last_error

    @staticmethod
    def _parse_batch(payload, expected):
        if not isinstance(payload, list) or len(payload) != expected:
            raise ValueError("Google вернул неверное количество переводов")
        values = []
        for item in payload:
            # With sl=auto, Chrome returns [translated text, detected language].
            value = item[0] if isinstance(item, list) and len(item) == 2 else item
            values.append(
                value.strip()
                if isinstance(value, str) and value.strip()
                else ValueError("Google вернул пустой или некорректный перевод")
            )
        return values

    @staticmethod
    def _parse(payload):
        if (
            not isinstance(payload, list)
            or not payload
            or not isinstance(payload[0], list)
        ):
            raise ValueError("Unexpected Google single response")
        values = []
        for segment in payload[0]:
            if (
                not isinstance(segment, list)
                or not segment
                or not isinstance(segment[0], str)
            ):
                raise ValueError("Unexpected Google translation segment")
            values.append(segment[0])
        result = "".join(values).strip()
        if not result:
            raise ValueError("Google returned empty text")
        return result
