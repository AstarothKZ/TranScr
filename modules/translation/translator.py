"""Translation manager: chooses a provider and keeps provider code independent."""

from collections import OrderedDict
from pathlib import Path

from .argos import ArgosTranslateProvider
from .google import GoogleTranslateProvider
from .yandex import YandexTranslateProvider

try:
    from dotenv import load_dotenv
except ImportError:  # .env is optional; environment variables are also supported.
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)


class Translator:
    """Stable facade used by the application; OCR and capture know nothing about providers."""

    PROVIDERS = {
        "google": "Google Translate",
        "yandex": "Yandex Cloud Translate",
        "argos": "Argos Translate (офлайн)",
        "hybrid": "Гибридный режим (Google → Argos)",
    }
    CACHE_SIZE = 64

    def __init__(
        self,
        from_code: str = "en",
        to_code: str = "ru",
        root=None,
        provider: str = "google",
        timeout: float = 15.0,
    ) -> None:
        if provider not in self.PROVIDERS:
            raise ValueError(f"Неизвестный переводчик: {provider}")
        self.from_code = from_code
        self.to_code = to_code
        self.root = root
        self.provider = provider
        self.timeout = timeout
        self._cache: OrderedDict[tuple[str, str, str, str], str] = OrderedDict()
        self._google_provider = None
        self._yandex_provider = None
        self._argos_provider = None

    def translate_text(self, text: str, provider: str | None = None) -> str:
        """Translate to the configured target language; cache repeated OCR blocks."""
        text = text.strip()
        if not text:
            return ""
        selected = provider or self.provider
        if selected not in self.PROVIDERS:
            raise ValueError(f"Неизвестный переводчик: {selected}")

        key = (selected, self.from_code, self.to_code, text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        translated = self._translate_with_provider(text, selected)
        self._cache[key] = translated
        self._cache.move_to_end(key)
        while len(self._cache) > self.CACHE_SIZE:
            self._cache.popitem(last=False)
        return translated

    def _translate_with_provider(self, text: str, selected: str) -> str:
        if selected == "google":
            return self._google().translate(text)
        if selected == "yandex":
            return self._yandex().translate(text)
        if selected == "argos":
            return self._argos().translate(text)

        # Hybrid: Google first, then the local Argos model if Google fails.
        try:
            return self._google().translate(text)
        except Exception as online_error:
            try:
                result = self._argos().translate(text)
                print(
                    "Google Translate недоступен; использован Argos. "
                    f"Причина: {type(online_error).__name__}",
                    flush=True,
                )
                return result
            except Exception as offline_error:
                raise RuntimeError(
                    "Не удалось перевести текст через Google, а офлайн-перевод "
                    "Argos недоступен. Проверьте интернет и установку модели Argos. "
                    f"Google: {online_error}; Argos: {offline_error}"
                ) from offline_error

    def _google(self) -> GoogleTranslateProvider:
        if self._google_provider is None:
            self._google_provider = GoogleTranslateProvider(
                self.from_code, self.to_code, self.timeout
            )
        return self._google_provider

    def _yandex(self) -> YandexTranslateProvider:
        if self._yandex_provider is None:
            self._yandex_provider = YandexTranslateProvider(
                self.from_code, self.to_code, self.timeout
            )
        return self._yandex_provider

    def _argos(self) -> ArgosTranslateProvider:
        if self._argos_provider is None:
            self._argos_provider = ArgosTranslateProvider(
                self.from_code, self.to_code, self.root
            )
        return self._argos_provider

    def close(self) -> None:
        """Close resources opened by translation providers."""
        for provider in (self._google_provider, self._yandex_provider, self._argos_provider):
            close = getattr(provider, "close", None)
            if callable(close):
                close()

    def check_language(self) -> bool:
        """Return True when the Argos model for the selected language pair is installed."""
        return self._argos().check_language()

    def manual_model_installation(self) -> bool:
        """Offer the user to install an Argos model from a local .argosmodel file."""
        return self._argos().manual_model_installation()
