"""Provider contract and ownership of bounded HTTP resources."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from time import perf_counter
from typing import Protocol, TypeAlias

from .batching import RequestBudget, make_client, validate_timeout

TranslationValue: TypeAlias = str | Exception


class TranslationProvider(Protocol):
    def translate_many(
        self,
        texts: list[str],
        *,
        cancel: Event | None = None,
        deadline: float | None = None,
    ) -> list[TranslationValue]: ...

    def close(self) -> None: ...


class NetworkProvider:
    """One pool per service; closing prevents queued work from starting requests."""

    MAX_RESPONSE_BYTES = 2 * 1024 * 1024

    def __init__(self, timeout: float, workers: int, name: str, **client_options):
        self.timeout = validate_timeout(timeout)
        self._stop = Event()
        self._lifecycle_lock = Lock()
        self._client = make_client(self.timeout, **client_options)
        self._executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix=name
        )

    def _budget(self, cancel=None, deadline=None) -> RequestBudget:
        if self._stop.is_set():
            raise RuntimeError("Переводчик уже закрыт")
        own_deadline = perf_counter() + self.timeout
        return RequestBudget(
            cancel,
            min(deadline, own_deadline) if deadline is not None else own_deadline,
            self._stop,
        )

    def translate(self, text: str) -> str:
        result = self.translate_many([text])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def _post_json(self, url: str, budget: RequestBudget, **kwargs):
        # Read timeouts apply per network chunk. Check the frame deadline while
        # streaming too, so a trickling response cannot occupy a worker forever.
        budget.check()
        with self._client.stream(
            "POST", url, timeout=budget.timeout(self.timeout), **kwargs
        ) as response:
            response.raise_for_status()
            body = bytearray()
            for chunk in response.iter_bytes():
                budget.check()
                if len(body) + len(chunk) > self.MAX_RESPONSE_BYTES:
                    raise ValueError("Ответ переводчика превышает допустимый размер")
                body.extend(chunk)
            budget.check()
            value = json.loads(body)
            budget.check()
            return value

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._stop.is_set():
                return
            self._stop.set()
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        finally:
            self._client.close()
