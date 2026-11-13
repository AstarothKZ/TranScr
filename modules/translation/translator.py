"""Batch-first translation facade with a frame budget, fallback and bounded cache."""

import logging
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, RLock
from time import perf_counter

from dotenv import load_dotenv

from modules.text_processing.filters import untranslated_fragments

from .argos import ArgosTranslateProvider
from .base import TranslationProvider
from .batching import RequestBudget, TranslationCancelled, validate_timeout
from .catalog import PROVIDER_LABELS
from .google import GoogleTranslateProvider
from .yandex import YandexTranslateProvider

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
LOGGER = logging.getLogger("transcr.pipeline")


def normalize_text(text: str) -> str:
    # Line geometry belongs to TextRegion. A translation request represents a
    # logical paragraph, not the OCR's arbitrary visual line breaks.
    text = re.sub("\u00ad\\s*\\n\\s*", "", text)
    return " ".join(text.split())


@dataclass
class TranslationStats:
    unique_texts: int = 0
    cache_hits: int = 0
    requests: int = 0
    incomplete: int = 0
    expires_at: float | None = None
    fallback_texts: int = 0
    errors: int = 0
    elapsed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class CacheEntry:
    text: str
    expires_at: float | None = None
    incomplete: bool = False
    fallback_used: bool = False


class Translator:
    PROVIDERS = PROVIDER_LABELS
    CACHE_SIZE = 2048
    CACHE_CHARS = 1_000_000
    PARTIAL_CACHE_TTL = 30.0
    FALLBACK_CACHE_TTL = 30.0

    def __init__(
        self,
        from_code="en",
        to_code="ru",
        root=None,
        provider="google",
        timeout=15.0,
        hybrid_enabled=False,
        fallback_provider="argos",
        repair_incomplete=False,
    ):
        timeout = validate_timeout(timeout)
        if (
            not isinstance(from_code, str)
            or not from_code.strip()
            or not isinstance(to_code, str)
            or not to_code.strip()
        ):
            raise ValueError("Коды языков должны быть непустыми строками")
        self._validate(provider, fallback_provider)
        if provider == fallback_provider:
            fallback_provider = "argos" if provider != "argos" else "google"
        self.from_code, self.to_code, self.root = (
            from_code.strip(),
            to_code.strip(),
            root,
        )
        self.provider, self.timeout = provider, timeout
        self.hybrid_enabled, self.fallback_provider = (
            bool(hybrid_enabled),
            fallback_provider,
        )
        self.repair_incomplete = bool(repair_incomplete)
        self._cache = OrderedDict()
        self._cache_chars = 0
        self._cache_lock, self._provider_lock = RLock(), RLock()
        self._call_lock = Lock()
        self._stop = Event()
        self._providers: dict[str, TranslationProvider] = {}
        self.last_stats = TranslationStats()

    @classmethod
    def _validate(cls, selected, fallback):
        for value in (selected, fallback):
            if not isinstance(value, str) or value not in cls.PROVIDERS:
                raise ValueError(f"Неизвестный переводчик: {value}")

    def _cache_key(self, text, selected=None, hybrid=None, fallback=None):
        return (
            selected or self.provider,
            self.from_code,
            self.to_code,
            text,
            self.hybrid_enabled if hybrid is None else hybrid,
            fallback or self.fallback_provider,
            self.repair_incomplete,
        )

    def _remember(self, key, value, *, fallback_used=False):
        incomplete = (
            bool(untranslated_fragments(value))
            if self.from_code == "en" and self.to_code == "ru"
            else False
        )
        ttl = min(
            self.PARTIAL_CACHE_TTL if incomplete else float("inf"),
            self.FALLBACK_CACHE_TTL if fallback_used else float("inf"),
        )
        expires_at = perf_counter() + ttl if ttl != float("inf") else None
        entry = CacheEntry(value, expires_at, incomplete, fallback_used)
        with self._cache_lock:
            if not value.strip() or self._stop.is_set():
                return entry
            previous = self._cache.pop(key, None)
            if previous is not None:
                self._cache_chars -= len(key[3]) + len(previous.text)
            self._cache[key] = entry
            self._cache_chars += len(key[3]) + len(value)
            while (
                len(self._cache) > self.CACHE_SIZE
                or self._cache_chars > self.CACHE_CHARS
            ):
                old_key, old = self._cache.popitem(last=False)
                self._cache_chars -= len(old_key[3]) + len(old.text)
        return entry

    def translate_many(
        self,
        texts,
        cancel: Event | None = None,
        *,
        provider=None,
        hybrid_enabled=None,
        fallback_provider=None,
    ):
        started = perf_counter()
        if self._stop.is_set():
            raise RuntimeError("Переводчик уже закрыт")
        selected = self.provider if provider is None else provider
        use_hybrid = (
            self.hybrid_enabled if hybrid_enabled is None else bool(hybrid_enabled)
        )
        fallback = (
            self.fallback_provider if fallback_provider is None else fallback_provider
        )
        self._validate(selected, fallback)
        use_hybrid = use_hybrid and selected != fallback
        budget = RequestBudget(cancel, started + self.timeout, self._stop)
        if isinstance(texts, (str, bytes)):
            raise TypeError("Ожидается последовательность строк, а не одна строка")
        originals = []
        for text in texts:
            if not isinstance(text, str):
                raise TypeError("Каждый исходный текст должен быть строкой")
            if text.strip():
                originals.append(text.strip())
        originals = list(dict.fromkeys(originals))
        normalized = {source: normalize_text(source) for source in originals}
        unique = list(dict.fromkeys(normalized.values()))
        stats = TranslationStats(unique_texts=len(unique))
        acquired = False
        try:
            budget.check()
            while not self._call_lock.acquire(timeout=0.05):
                budget.check()
            acquired = True
            budget.check()
            return self._translate_many(
                originals,
                normalized,
                unique,
                selected,
                use_hybrid,
                fallback,
                budget,
                stats,
                started,
            )
        except (TranslationCancelled, TimeoutError) as error:
            stats.errors = len(unique)
            stats.elapsed_ms = (perf_counter() - started) * 1000
            self.last_stats = stats
            return {source: error for source in originals}
        finally:
            if acquired:
                self._call_lock.release()

    def _translate_many(
        self,
        originals,
        normalized,
        unique,
        selected,
        use_hybrid,
        fallback,
        budget,
        stats,
        started,
    ):
        values, missing = {}, []
        for text in unique:
            key = self._cache_key(text, selected, use_hybrid, fallback)
            with self._cache_lock:
                if key in self._cache:
                    entry = self._cache[key]
                    if entry.expires_at is None or perf_counter() < entry.expires_at:
                        values[text] = entry.text
                        self._cache.move_to_end(key)
                        stats.cache_hits += 1
                        if entry.expires_at is not None:
                            stats.expires_at = min(
                                stats.expires_at or entry.expires_at, entry.expires_at
                            )
                        stats.incomplete += int(entry.incomplete)
                        stats.fallback_texts += int(entry.fallback_used)
                        continue
                    self._cache.pop(key)
                    self._cache_chars -= len(key[3]) + len(entry.text)
                missing.append(text)
        requests_before = self._request_count()
        if missing:
            primary_budget = RequestBudget(
                budget.cancel,
                budget.deadline - min(5.0, self.timeout / 3)
                if use_hybrid
                else budget.deadline,
                budget.stop,
            )
            primary = self._provider_many(selected, missing, primary_budget)
            values.update(zip(missing, primary))
            failed = [
                text
                for text in missing
                if isinstance(values[text], Exception)
                and not isinstance(values[text], TranslationCancelled)
            ]
            fallback_used = set()
            if use_hybrid and failed:
                reserve = self._provider_many(fallback, failed, budget)
                for text, result in zip(failed, reserve):
                    if isinstance(result, Exception):
                        values[text] = RuntimeError(
                            f"Основной переводчик: {values[text]}; резервный: {result}"
                        )
                    else:
                        values[text] = result
                        fallback_used.add(text)
            if (
                self.repair_incomplete
                and self.from_code == "en"
                and self.to_code == "ru"
            ):
                self._repair(values, missing, selected, fallback, use_hybrid, budget)
            for text in missing:
                value = values[text]
                if isinstance(value, str):
                    entry = self._remember(
                        self._cache_key(text, selected, use_hybrid, fallback),
                        value,
                        fallback_used=text in fallback_used,
                    )
                    stats.incomplete += int(entry.incomplete)
                    stats.fallback_texts += int(entry.fallback_used)
                    if entry.expires_at is not None:
                        stats.expires_at = min(
                            stats.expires_at or entry.expires_at, entry.expires_at
                        )
        stats.requests = self._request_count() - requests_before
        stats.errors = sum(isinstance(value, Exception) for value in values.values())
        stats.elapsed_ms = (perf_counter() - started) * 1000
        self.last_stats = stats
        LOGGER.info(
            "Перевод: %.0f мс; HTTP %d; кэш %d/%d; резерв %d; неполных %d; ошибок %d",
            stats.elapsed_ms,
            stats.requests,
            stats.cache_hits,
            stats.unique_texts,
            stats.fallback_texts,
            stats.incomplete,
            stats.errors,
        )
        return {source: values[normalized[source]] for source in originals}

    def _provider_many(self, selected, texts, budget):
        try:
            budget.check()
            provider = self._get_provider(selected)
            values = provider.translate_many(
                texts, cancel=budget.cancel, deadline=budget.deadline
            )
            if not isinstance(values, list) or len(values) != len(texts):
                raise ValueError("Переводчик вернул неверное количество результатов")
            return [
                value
                if isinstance(value, Exception)
                or (isinstance(value, str) and value.strip())
                else ValueError("Переводчик вернул пустой результат")
                for value in values
            ]
        except Exception as error:
            return [error] * len(texts)

    def _repair(self, values, texts, selected, fallback, use_hybrid, budget):
        # One additional batch, not up to three serial requests for every region.
        # Keep a separate short budget for this optional quality pass.
        repair_budget = RequestBudget(
            budget.cancel, min(budget.deadline, perf_counter() + 2), budget.stop
        )
        candidates = []
        for text in texts:
            if not isinstance(values[text], str):
                continue
            for start, end in untranslated_fragments(values[text])[:2]:
                candidates.append((text, start, end, values[text][start:end]))
        if not candidates:
            return
        fragments = list(dict.fromkeys(item[3] for item in candidates))
        translated = dict(
            zip(
                fragments,
                self._provider_many(
                    fallback if use_hybrid else selected, fragments, repair_budget
                ),
            )
        )
        for text, start, end, fragment in reversed(candidates):
            replacement = translated[fragment]
            if isinstance(replacement, str) and re.search("[А-Яа-яЁё]", replacement):
                values[text] = (
                    values[text][:start] + replacement.strip() + values[text][end:]
                )

    def translate_text(
        self, text, provider=None, *, hybrid_enabled=None, fallback_provider=None
    ):
        if not isinstance(text, str):
            raise TypeError("Исходный текст должен быть строкой")
        source = text.strip()
        if not source:
            return ""
        value = self.translate_many(
            [source],
            provider=provider,
            hybrid_enabled=hybrid_enabled,
            fallback_provider=fallback_provider,
        )[source]
        if isinstance(value, Exception):
            raise value
        return value

    def _get_provider(self, selected):
        constructors = {
            "google": GoogleTranslateProvider,
            "yandex": YandexTranslateProvider,
            "argos": ArgosTranslateProvider,
        }
        with self._provider_lock:
            if self._stop.is_set():
                raise RuntimeError("Переводчик уже закрыт")
            instance = self._providers.get(selected)
            if instance is None:
                if selected == "argos":
                    instance = constructors[selected](
                        self.from_code, self.to_code, self.root, timeout=self.timeout
                    )
                else:
                    instance = constructors[selected](
                        self.from_code, self.to_code, self.timeout
                    )
                self._providers[selected] = instance
            return instance

    def _request_count(self):
        with self._provider_lock:
            instances = list(self._providers.values())
        return sum(
            getattr(
                getattr(instance, "_client", None),
                "transcr_requests",
                0,
            )
            for instance in instances
        )

    def check_language(self):
        return self._get_provider("argos").check_language()

    def warmup(self):
        instance = self._get_provider(self.provider)
        if self.provider == "argos":
            instance.warmup()

    def manual_model_installation(self):
        return self._get_provider("argos").manual_model_installation()

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()
            self._cache_chars = 0

    def close(self):
        with self._provider_lock:
            if self._stop.is_set():
                return
            self._stop.set()
            instances = list(self._providers.items())
        for name, instance in instances:
            try:
                instance.close()
            except Exception:
                LOGGER.exception("Не удалось закрыть переводчик %s", name)
        self.clear_cache()
