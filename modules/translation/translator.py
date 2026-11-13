"""Batch-first translation facade with a frame budget, fallback and bounded cache."""

from collections import OrderedDict
from dataclasses import dataclass
import logging
from pathlib import Path
import re
from threading import Event, RLock
from time import perf_counter

from dotenv import load_dotenv

from modules.text_processing.filters import untranslated_fragments
from .argos import ArgosTranslateProvider
from .batching import RequestBudget, TranslationCancelled
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


@dataclass(frozen=True, slots=True)
class CacheEntry:
    text: str
    expires_at: float | None = None


class Translator:
    PROVIDERS = PROVIDER_LABELS
    CACHE_SIZE = 2048
    CACHE_CHARS = 1_000_000
    PARTIAL_CACHE_TTL = 30.0

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
        if timeout <= 0:
            raise ValueError("Время перевода должно быть положительным")
        self._validate(provider, fallback_provider)
        if provider == fallback_provider:
            fallback_provider = "argos" if provider != "argos" else "google"
        self.from_code, self.to_code, self.root = from_code, to_code, root
        self.provider, self.timeout = provider, timeout
        self.hybrid_enabled, self.fallback_provider = (
            bool(hybrid_enabled),
            fallback_provider,
        )
        self.repair_incomplete = bool(repair_incomplete)
        self._cache = OrderedDict()
        self._cache_chars = 0
        self._cache_lock, self._provider_lock = RLock(), RLock()
        self._google_provider = self._yandex_provider = self._argos_provider = None
        self.last_stats = TranslationStats()
        self._closed = False

    @classmethod
    def _validate(cls, selected, fallback):
        for value in (selected, fallback):
            if value not in cls.PROVIDERS:
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

    def _remember(self, key, value):
        if not value.strip() or self._closed:
            return None
        expires_at = (
            perf_counter() + self.PARTIAL_CACHE_TTL
            if untranslated_fragments(value)
            else None
        )
        entry = CacheEntry(value, expires_at)
        with self._cache_lock:
            if self._closed:
                return None
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
        return expires_at

    def translate_many(
        self,
        texts,
        cancel: Event | None = None,
        *,
        provider=None,
        hybrid_enabled=None,
        fallback_provider=None,
    ):
        if self._closed:
            raise RuntimeError("Переводчик уже закрыт")
        selected = provider or self.provider
        use_hybrid = (
            self.hybrid_enabled if hybrid_enabled is None else bool(hybrid_enabled)
        )
        fallback = fallback_provider or self.fallback_provider
        self._validate(selected, fallback)
        use_hybrid = use_hybrid and selected != fallback
        budget = RequestBudget(cancel, perf_counter() + self.timeout)
        originals = list(dict.fromkeys(text.strip() for text in texts if text.strip()))
        normalized = {source: normalize_text(source) for source in originals}
        unique = list(dict.fromkeys(normalized.values()))
        stats = TranslationStats(unique_texts=len(unique))
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
                            stats.incomplete += 1
                        continue
                    self._cache.pop(key)
                    self._cache_chars -= len(key[3]) + len(entry.text)
                missing.append(text)
        requests_before = self._request_count()
        if missing:
            primary = self._provider_many(selected, missing, budget)
            values.update(zip(missing, primary))
            failed = [
                text
                for text in missing
                if isinstance(values[text], Exception)
                and not isinstance(values[text], (TranslationCancelled, TimeoutError))
            ]
            if use_hybrid and failed:
                reserve = self._provider_many(fallback, failed, budget)
                for text, result in zip(failed, reserve):
                    if isinstance(result, Exception):
                        values[text] = RuntimeError(
                            f"Основной переводчик: {values[text]}; резервный: {result}"
                        )
                    else:
                        values[text] = result
            if (
                self.repair_incomplete
                and self.from_code == "en"
                and self.to_code == "ru"
            ):
                self._repair(values, missing, selected, fallback, use_hybrid, budget)
            for text in missing:
                value = values[text]
                if isinstance(value, str):
                    if untranslated_fragments(value):
                        stats.incomplete += 1
                    expires_at = self._remember(
                        self._cache_key(text, selected, use_hybrid, fallback), value
                    )
                    if expires_at is not None:
                        stats.expires_at = min(
                            stats.expires_at or expires_at, expires_at
                        )
        stats.requests = self._request_count() - requests_before
        self.last_stats = stats
        LOGGER.info(
            "Перевод: HTTP %d; кэш %d/%d; неполных %d",
            stats.requests,
            stats.cache_hits,
            stats.unique_texts,
            stats.incomplete,
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
            budget.cancel, min(budget.deadline, perf_counter() + 2)
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
        field = f"_{selected}_provider"
        with self._provider_lock:
            if self._closed:
                raise RuntimeError("Переводчик уже закрыт")
            instance = getattr(self, field)
            if instance is None:
                if selected == "argos":
                    instance = constructors[selected](
                        self.from_code, self.to_code, self.root
                    )
                else:
                    instance = constructors[selected](
                        self.from_code, self.to_code, self.timeout
                    )
                setattr(self, field, instance)
            return instance

    def _request_count(self):
        return sum(
            getattr(
                getattr(getattr(self, f"_{name}_provider"), "_client", None),
                "transcr_requests",
                0,
            )
            for name in self.PROVIDERS
        )

    def check_language(self):
        return self._get_provider("argos").check_language()

    def warmup(self):
        if self.provider == "argos":
            self._get_provider("argos").warmup()

    def manual_model_installation(self):
        return self._get_provider("argos").manual_model_installation()

    def close(self):
        with self._provider_lock:
            if self._closed:
                return
            self._closed = True
        for name in self.PROVIDERS:
            instance = getattr(self, f"_{name}_provider")
            if instance is not None:
                try:
                    instance.close()
                except Exception:
                    LOGGER.exception("Не удалось закрыть переводчик %s", name)
        with self._cache_lock:
            self._cache.clear()
            self._cache_chars = 0
