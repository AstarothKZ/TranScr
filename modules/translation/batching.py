"""Bounded translation jobs, stable chunk mapping and a shared request budget."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from importlib.util import find_spec
from threading import Event
from threading import Lock
import logging
import os
import math
from time import perf_counter
from typing import Callable, TypeVar

import httpx

from .text_chunks import split_text

T = TypeVar("T")
R = TypeVar("R")


class TranslationCancelled(RuntimeError):
    pass


class ProviderUnavailable(RuntimeError):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def service_unavailable(error: Exception) -> bool:
    cause = error
    while cause is not None:
        if isinstance(cause, ProviderUnavailable):
            return True
        if isinstance(cause, httpx.TransportError):
            return True
        if isinstance(cause, httpx.HTTPStatusError) and (
            cause.response.status_code in {401, 403, 429}
            or cause.response.status_code >= 500
        ):
            return True
        cause = cause.__cause__
    return False


def retry_delay(value: str | None, minimum: float) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return minimum
    return max(minimum, min(86400, seconds)) if math.isfinite(seconds) else minimum


class ServiceCooldown:
    """Share a service failure between captures, without sleeping any worker."""

    def __init__(self, name: str):
        self.name = name
        self._lock = Lock()
        self._until = 0.0
        self._error = None

    def check(self):
        with self._lock:
            if perf_counter() < self._until:
                raise ProviderUnavailable(
                    f"{self.name}: пауза после ошибки сервиса"
                ) from self._error
            self._error = None

    def record(self, error):
        if not service_unavailable(error):
            return
        minimum, header = 5.0, None
        cause = error
        while cause is not None:
            if isinstance(cause, ProviderUnavailable) and cause.status in {
                401,
                403,
                429,
            }:
                minimum = 30.0
            if isinstance(cause, httpx.HTTPStatusError):
                if cause.response.status_code in {401, 403, 429}:
                    minimum = 30.0
                header = cause.response.headers.get("Retry-After")
            cause = cause.__cause__
        with self._lock:
            self._until = max(
                self._until, perf_counter() + retry_delay(header, minimum)
            )
            self._error = error


@dataclass(frozen=True)
class RequestBudget:
    cancel: Event | None = None
    deadline: float | None = None

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise TranslationCancelled("Перевод отменён")
        if self.deadline is not None and perf_counter() >= self.deadline:
            raise TimeoutError("Истекло время перевода кадра")

    def timeout(self, timeout: float) -> httpx.Timeout:
        self.check()
        remaining = (
            timeout
            if self.deadline is None
            else min(timeout, self.deadline - perf_counter())
        )
        remaining = max(0.05, remaining)
        return httpx.Timeout(
            min(8.0, remaining),
            connect=min(3.0, remaining),
            write=min(3.0, remaining),
            pool=min(1.0, remaining),
        )


def make_client(timeout: float, **kwargs) -> httpx.Client:
    lock = Lock()

    def started(request):
        request.extensions["transcr_started"] = perf_counter()
        with lock:
            client.transcr_requests += 1

    def completed(response):
        elapsed = (
            perf_counter() - response.request.extensions["transcr_started"]
        ) * 1000
        logging.getLogger("transcr.pipeline").info(
            "%s HTTP %s %s: %.0f мс",
            response.request.url.host,
            response.status_code,
            response.http_version,
            elapsed,
        )

    client = httpx.Client(
        timeout=RequestBudget().timeout(timeout),
        http2=find_spec("h2") is not None and os.getenv("TRANSCR_HTTP2", "1") != "0",
        limits=httpx.Limits(
            max_connections=4, max_keepalive_connections=4, keepalive_expiry=60
        ),
        **kwargs,
    )
    client.transcr_requests = 0
    client.event_hooks["request"].append(started)
    client.event_hooks["response"].append(completed)
    return client


@dataclass(frozen=True)
class TextPart:
    owner: int
    text: str


def text_parts(texts: list[str], limit: int, *, utf8_bytes=False) -> list[TextPart]:
    return [
        TextPart(index, part)
        for index, text in enumerate(texts)
        for part in split_text(text, limit, utf8_bytes=utf8_bytes)
    ]


def pack_parts(
    parts: list[TextPart], limit: int, max_items: int = 128
) -> list[list[TextPart]]:
    if limit < 1 or max_items < 1:
        raise ValueError("Лимиты пакета должны быть положительными")
    batches, batch, size = [], [], 0
    for part in parts:
        if batch and (size + len(part.text) > limit or len(batch) >= max_items):
            batches.append(batch)
            batch, size = [], 0
        batch.append(part)
        size += len(part.text)
    if batch:
        batches.append(batch)
    return batches


def join_parts(
    texts: list[str], parts: list[TextPart], values: list[str | Exception]
) -> list[str | Exception]:
    if len(parts) != len(values):
        raise ValueError("Неверное количество результатов перевода")
    groups: list[list[str | Exception]] = [[] for _ in texts]
    for part, value in zip(parts, values):
        groups[part.owner].append(value)
    return [
        next((v for v in group if isinstance(v, Exception)), None) or "\n".join(group)
        for group in groups
    ]


def map_bounded(
    executor: ThreadPoolExecutor,
    function: Callable[[T], R],
    items: list[T],
    budget: RequestBudget,
    workers: int,
) -> list[R | Exception]:
    """Never enqueue a whole frame; cancellation returns without waiting on its tail."""
    results: list[R | Exception] = [
        TranslationCancelled("Перевод не выполнен") for _ in items
    ]
    pending = {}
    position = 0
    try:
        while position < len(items) or pending:
            budget.check()
            while position < len(items) and len(pending) < workers:
                budget.check()
                future = executor.submit(function, items[position])
                pending[future] = position
                position += 1
            done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
            for future in done:
                index = pending.pop(future)
                try:
                    results[index] = future.result()
                except Exception as error:
                    results[index] = error
    except (TranslationCancelled, TimeoutError) as error:
        for future, index in pending.items():
            future.cancel()
            results[index] = error
        for index in range(position, len(items)):
            results[index] = error
    return results


def batch_results(
    texts: list[str],
    parts: list[TextPart],
    batches: list[list[TextPart]],
    values: list[list[str | Exception] | Exception],
) -> list[str | Exception]:
    if len(batches) != len(values):
        raise ValueError("Неверное количество ответов на пакеты")
    flat = []
    for batch, value in zip(batches, values):
        if isinstance(value, Exception):
            flat.extend([value] * len(batch))
        elif not isinstance(value, list) or len(value) != len(batch):
            flat.extend(
                [ValueError("Неверное количество переводов в пакете")] * len(batch)
            )
        else:
            flat.extend(value)
    return join_parts(texts, parts, flat)
