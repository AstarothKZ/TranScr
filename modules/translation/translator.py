"""Translation manager: provider selection, hybrid fallback and memory cache."""

import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, RLock

import httpx

from modules.text_processing.filters import untranslated_fragments

from .argos import ArgosTranslateProvider
from .catalog import PROVIDER_LABELS
from .google import GoogleTranslateError, GoogleTranslateProvider
from .mymemory import MyMemoryTranslateProvider
from .yandex import YandexTranslateProvider

try:
    from dotenv import load_dotenv
except ImportError:  # .env is optional; environment variables are also supported.
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


class Translator:
    """Stable facade; primary and fallback providers can be chosen independently."""

    PROVIDERS = PROVIDER_LABELS
    CACHE_SIZE = 128

    def __init__(
        self,
        from_code: str = "en",
        to_code: str = "ru",
        root=None,
        provider: str = "mymemory",
        timeout: float = 15.0,
        hybrid_enabled: bool = False,
        fallback_provider: str = "argos",
    ) -> None:
        # Accept the previous API's provider="hybrid" for config scripts/tests.
        if provider == "hybrid":
            provider = "mymemory"
            hybrid_enabled = True
            fallback_provider = "argos"
        if provider not in self.PROVIDERS:
            raise ValueError(f"Неизвестный переводчик: {provider}")
        if fallback_provider not in self.PROVIDERS:
            raise ValueError(f"Неизвестный резервный переводчик: {fallback_provider}")
        if provider == fallback_provider:
            fallback_provider = "argos" if provider != "argos" else "mymemory"

        self.from_code = from_code
        self.to_code = to_code
        self.root = root
        self.provider = provider
        self.timeout = timeout
        self.hybrid_enabled = bool(hybrid_enabled)
        self.fallback_provider = fallback_provider
        self._cache: OrderedDict[tuple[str, str, str, str, bool, str], str] = (
            OrderedDict()
        )
        self._cache_lock = RLock()
        self._provider_lock = RLock()
        self._argos_translation_lock = RLock()
        self._mymemory_provider = None
        self._google_provider = None
        self._yandex_provider = None
        self._argos_provider = None
        self._executor = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="transcr-translate"
        )

    def translate_many(
        self, texts: list[str], cancel: Event | None = None
    ) -> dict[str, str | Exception]:
        """Translate a frame with one scheduling layer and per-region failures.

        A failed transport disables further primary requests for this frame only;
        the next capture can retry. Already running requests finish normally.
        """
        texts = list(dict.fromkeys(text.strip() for text in texts if text.strip()))
        results: dict[str, str | Exception] = {}
        missing = []
        for text in texts:
            key = self._cache_key(text)
            with self._cache_lock:
                if key in self._cache:
                    results[text] = self._cache[key]
                    self._cache.move_to_end(key)
                else:
                    missing.append(text)
        if not missing or (cancel is not None and cancel.is_set()):
            return results

        transport_failed = Event()

        def fallback_one(text: str, primary_error: Exception) -> str | Exception:
            if not self.hybrid_enabled:
                return primary_error
            if cancel is not None and cancel.is_set():
                return RuntimeError("Перевод отменён")
            try:
                return self.translate_text(
                    text, provider=self.fallback_provider, hybrid_enabled=False
                )
            except Exception as fallback_error:
                return RuntimeError(
                    f"Основной переводчик: {primary_error}; резервный: {fallback_error}"
                )

        def translate_one(text: str) -> str | Exception:
            if cancel is not None and cancel.is_set():
                return RuntimeError("Перевод отменён")
            if transport_failed.is_set():
                return fallback_one(
                    text, RuntimeError("Сервис недоступен; запрос этого кадра пропущен")
                )
            try:
                value = self.translate_text(text, hybrid_enabled=False)
                self._remember(self._cache_key(text), value)
                return value
            except Exception as error:
                cause = error
                while cause is not None:
                    if isinstance(cause, (httpx.TransportError, GoogleTranslateError)) or (
                        isinstance(cause, httpx.HTTPStatusError)
                        and (
                            cause.response.status_code in {401, 403, 429}
                            or cause.response.status_code >= 500
                        )
                    ):
                        transport_failed.set()
                        break
                    cause = cause.__cause__
                return fallback_one(text, error)

        if self.provider == "yandex":
            try:
                translated = self._yandex().translate_many(missing)
                for text, value in zip(missing, translated):
                    try:
                        value = self._complete_translation(value, self.provider, text)
                        results[text] = value
                        self._remember(self._cache_key(text), value)
                    except Exception as error:
                        results[text] = fallback_one(text, error)
                return results
            except Exception as error:
                if not self.hybrid_enabled:
                    results.update((text, error) for text in missing)
                    return results
                # Do not repeat the failed batch as individual primary requests.
                transport_failed.set()

        if self.provider == "argos":
            for text in missing:
                results[text] = translate_one(text)
        else:
            futures = {
                text: self._executor.submit(translate_one, text) for text in missing
            }
            results.update((text, future.result()) for text, future in futures.items())
        return results

    def _cache_key(self, text: str):
        return (
            self.provider,
            self.from_code,
            self.to_code,
            text,
            self.hybrid_enabled,
            self.fallback_provider,
        )

    def _remember(self, key, translated: str) -> None:
        if untranslated_fragments(translated):
            return
        with self._cache_lock:
            self._cache[key] = translated
            self._cache.move_to_end(key)
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)

    def translate_text(
        self,
        text: str,
        provider: str | None = None,
        *,
        hybrid_enabled: bool | None = None,
        fallback_provider: str | None = None,
    ) -> str:
        """Translate text; the optional fallback is used only after an error."""
        text = text.strip()
        if not text:
            return ""

        selected = provider or self.provider
        use_hybrid = (
            self.hybrid_enabled if hybrid_enabled is None else bool(hybrid_enabled)
        )
        fallback = fallback_provider or self.fallback_provider
        if selected == "hybrid":  # Backwards-compatible call style.
            selected, use_hybrid, fallback = self.provider, True, self.fallback_provider
        if selected not in self.PROVIDERS:
            raise ValueError(f"Неизвестный переводчик: {selected}")
        if fallback not in self.PROVIDERS:
            raise ValueError(f"Неизвестный резервный переводчик: {fallback}")
        if selected == fallback:
            use_hybrid = False

        key = (selected, self.from_code, self.to_code, text, use_hybrid, fallback)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        translated = self._translate(text, selected, use_hybrid, fallback)
        self._remember(key, translated)
        return translated

    def _translate(
        self, text: str, selected: str, hybrid_enabled: bool, fallback: str
    ) -> str:
        if not hybrid_enabled:
            return self._translate_with_provider(text, selected)

        primary_error: Exception | None = None
        try:
            return self._translate_with_provider(text, selected)
        except Exception as error:
            primary_error = error

        try:
            return self._translate_with_provider(text, fallback)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Основной переводчик {selected} не сработал: {primary_error}. "
                f"Резервный переводчик {fallback} также не сработал: {fallback_error}"
            ) from fallback_error

    def _translate_with_provider(self, text: str, selected: str) -> str:
        return self._complete_translation(
            self._provider_translate(text, selected), selected, text
        )

    def _complete_translation(self, result: str, selected: str, source: str) -> str:
        if self.from_code != "en" or self.to_code != "ru":
            return result
        fragments = untranslated_fragments(result)
        for start, end in reversed(fragments[:2]):
            try:
                replacement = self._provider_translate(
                    result[start:end], selected
                ).strip()
            except Exception:
                continue
            if replacement and any(
                "а" <= char.lower() <= "я" or char.lower() == "ё"
                for char in replacement
            ):
                result = result[:start] + replacement + result[end:]
        if untranslated_fragments(result):
            if self.hybrid_enabled and selected != self.fallback_provider:
                try:
                    fallback = self._provider_translate(source, self.fallback_provider)
                    if fallback.strip() and not untranslated_fragments(fallback):
                        return fallback
                except Exception:
                    pass
            logging.getLogger("transcr.pipeline").warning(
                "Показан неполный перевод; проверьте распознавание исходника: %r",
                result,
            )
        return result

    def _provider_translate(self, text: str, selected: str) -> str:
        if selected == "google":
            return self._google().translate(text)
        if selected == "mymemory":
            return self._mymemory().translate(text)
        if selected == "yandex":
            return self._yandex().translate(text)
        if selected == "argos":
            # Keep a single in-process Argos operation at a time while allowing
            # online primary requests to run concurrently in hybrid mode.
            with self._argos_translation_lock:
                return self._argos().translate(text)
        raise ValueError(f"Неизвестный переводчик: {selected}")

    def _google(self) -> GoogleTranslateProvider:
        with self._provider_lock:
            if self._google_provider is None:
                self._google_provider = GoogleTranslateProvider(
                    self.from_code, self.to_code, self.timeout
                )
            return self._google_provider

    def _mymemory(self) -> MyMemoryTranslateProvider:
        with self._provider_lock:
            if self._mymemory_provider is None:
                self._mymemory_provider = MyMemoryTranslateProvider(
                    self.from_code, self.to_code, self.timeout
                )
            return self._mymemory_provider

    def _yandex(self) -> YandexTranslateProvider:
        with self._provider_lock:
            if self._yandex_provider is None:
                self._yandex_provider = YandexTranslateProvider(
                    self.from_code, self.to_code, self.timeout
                )
            return self._yandex_provider

    def _argos(self) -> ArgosTranslateProvider:
        with self._provider_lock:
            if self._argos_provider is None:
                self._argos_provider = ArgosTranslateProvider(
                    self.from_code, self.to_code, self.root
                )
            return self._argos_provider

    def close(self) -> None:
        """Close resources opened by translation providers."""
        self._executor.shutdown(wait=True, cancel_futures=True)
        for provider in (
            self._google_provider,
            self._mymemory_provider,
            self._yandex_provider,
            self._argos_provider,
        ):
            if provider is not None:
                try:
                    provider.close()
                except Exception:
                    pass

    def check_language(self) -> bool:
        """Return True when the Argos model for the selected language pair is installed."""
        return self._argos().check_language()

    def manual_model_installation(self) -> bool:
        """Offer the user to install an Argos model from a local .argosmodel file."""
        return self._argos().manual_model_installation()
