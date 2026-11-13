"""Argos Translate provider for offline translation."""

import os
import re
from pathlib import Path
from threading import Event, RLock
from time import perf_counter

from .batching import RequestBudget, TextPart, join_parts, validate_timeout
from .text_chunks import split_text

_SENTENCE_GAP = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ABBREVIATIONS = frozenset(
    {
        "mr.",
        "mrs.",
        "ms.",
        "dr.",
        "prof.",
        "capt.",
        "col.",
        "gen.",
        "lt.",
        "sgt.",
        "st.",
        "no.",
        "co.",
        "vs.",
        "e.g.",
        "i.e.",
    }
)


def sentence_parts(texts, language="en"):
    """Keep sentence boundaries and identifiers without loading an NLP model."""
    parts = []
    for owner, text in enumerate(texts):
        start = 0
        boundaries = _SENTENCE_GAP.finditer(text) if language == "en" else ()
        for boundary in boundaries:
            prefix = text[start : boundary.start()]
            previous = prefix.rsplit(maxsplit=1)[-1] if prefix.strip() else ""
            if previous.lower() in _ABBREVIATIONS or re.fullmatch(
                r"(?:[A-Z]\.)+", previous
            ):
                continue
            parts.extend(TextPart(owner, value) for value in split_text(prefix, 1000))
            start = boundary.end()
        parts.extend(TextPart(owner, value) for value in split_text(text[start:], 1000))
    return parts


class ArgosTranslateProvider:
    def __init__(
        self, from_code: str = "en", to_code: str = "ru", root=None, *, timeout=15.0
    ):
        self.from_code = from_code
        self.to_code = to_code
        self.root = root
        self.timeout = validate_timeout(timeout)
        self._package = None
        self._engine = None
        self._lock = RLock()
        self._stop = Event()

    def translate(self, text: str) -> str:
        result = self.translate_many([text])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def _find_package(self):
        import argostranslate.package as package

        return next(
            (
                item
                for item in package.get_installed_packages()
                if item.from_code == self.from_code and item.to_code == self.to_code
            ),
            None,
        )

    def _ensure_engine(self):
        if self._stop.is_set():
            raise RuntimeError("Переводчик уже закрыт")
        if self._engine is not None:
            return
        try:
            import ctranslate2
            from argostranslate import settings
        except ImportError as error:
            raise RuntimeError(
                "Argos не установлен. Установите requirements-optional.txt"
            ) from error
        self._package = self._find_package()
        if self._package is None:
            raise RuntimeError(
                f"В Argos не найдена модель {self.from_code} → {self.to_code}"
            )
        self._beam_size = int(settings.get_setting("ARGOS_BEAM_SIZE", "2"))
        compute = settings.get_setting(
            "ARGOS_COMPUTE_TYPE", "int8" if settings.device == "cpu" else "auto"
        )
        self._engine = ctranslate2.Translator(
            str(self._package.package_path / "model"),
            device=settings.device,
            compute_type=compute,
            inter_threads=int(settings.get_setting("ARGOS_INTER_THREADS", "1")),
            intra_threads=int(
                settings.get_setting(
                    "ARGOS_INTRA_THREADS", str(min(4, os.cpu_count() or 4))
                )
            ),
        )

    def translate_many(self, texts, *, cancel=None, deadline=None):
        if self._stop.is_set():
            raise RuntimeError("Переводчик уже закрыт")
        own_deadline = perf_counter() + self.timeout
        budget = RequestBudget(
            cancel,
            min(deadline, own_deadline) if deadline is not None else own_deadline,
            self._stop,
        )
        parts = sentence_parts(texts, self.from_code)
        if not parts:
            return [""] * len(texts)
        try:
            while not self._lock.acquire(timeout=0.05):
                budget.check()
            try:
                budget.check()
                self._ensure_engine()
                budget.check()
                tokens = [self._package.tokenizer.encode(part.text) for part in parts]
                prefix = self._package.target_prefix
                translated = self._engine.translate_batch(
                    tokens,
                    target_prefix=[[prefix]] * len(tokens) if prefix else None,
                    replace_unknowns=True,
                    max_batch_size=1024,
                    batch_type="tokens",
                    beam_size=max(1, self._beam_size),
                    num_hypotheses=1,
                    return_scores=False,
                    length_penalty=0.2,
                    max_input_length=0,
                    max_decoding_length=max(256, max(map(len, tokens)) * 4),
                )
                budget.check()
                values = []
                for result in translated:
                    try:
                        value = self._package.tokenizer.decode(result.hypotheses[0])
                        if prefix and value.startswith(prefix):
                            value = value[len(prefix) :]
                        values.append(value.strip())
                    except (IndexError, AttributeError, TypeError, ValueError) as error:
                        values.append(
                            ValueError(
                                f"Argos вернул некорректный сегмент: {type(error).__name__}"
                            )
                        )
                return join_parts(texts, parts, values)
            finally:
                self._lock.release()
        except Exception as error:
            return [error] * len(texts)

    def warmup(self):
        with self._lock:
            self._ensure_engine()

    def check_language(self) -> bool:
        try:
            return self._find_package() is not None
        except ImportError:
            return False

    def manual_model_installation(self) -> bool:
        from tkinter import filedialog, messagebox

        try:
            import argostranslate.package as argos_package
        except ImportError:
            print(
                "Argos Translate не установлен. Выполните: "
                "pip install -r requirements-optional.txt",
                flush=True,
            )
            return False

        if self.root is None:
            raise RuntimeError("Для выбора модели необходимо окно приложения")
        accepted = messagebox.askyesno(
            parent=self.root,
            title="Установка офлайн-переводчика",
            message="Выбрать файл языковой модели Argos (.argosmodel)?",
        )
        if not accepted:
            return False
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Выберите языковую модель Argos",
            filetypes=[("Языковые модели Argos", "*.argosmodel")],
            initialdir=str(Path.home() / "Downloads"),
        )
        if not path:
            return False
        try:
            with self._lock:
                if self._stop.is_set():
                    raise RuntimeError("Переводчик уже закрыт")
                argos_package.install_from_path(Path(path))
                if self._engine is not None:
                    self._engine.unload_model()
                self._engine = self._package = None
        except Exception as error:
            print(f"Не удалось установить модель Argos: {error}", flush=True)
            return False
        print("Модель Argos установлена.", flush=True)
        return self.check_language()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            engine, self._engine = self._engine, None
            self._package = None
            if engine is not None:
                engine.unload_model()
