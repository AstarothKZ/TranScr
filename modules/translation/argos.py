"""Argos Translate provider for offline translation."""

from pathlib import Path
from tkinter import filedialog, messagebox


class ArgosTranslateProvider:
    def __init__(self, from_code: str = "en", to_code: str = "ru", root=None):
        self.from_code = from_code
        self.to_code = to_code
        self.root = root

    def translate(self, text: str) -> str:
        argos_translate = self._get_argos_translate()
        installed_languages = argos_translate.get_installed_languages()
        source = next(
            (language for language in installed_languages if language.code == self.from_code),
            None,
        )
        if source is None:
            raise RuntimeError(f"В Argos не установлен язык: {self.from_code}")

        translation = next(
            (item for item in source.translations_from if item.to_lang.code == self.to_code),
            None,
        )
        if translation is None:
            raise RuntimeError(
                f"В Argos не найдена модель {self.from_code} → {self.to_code}. "
                "Установите соответствующий языковой пакет."
            )
        return translation.translate(text)

    def check_language(self) -> bool:
        try:
            import argostranslate.translate as argos_translate
        except ImportError:
            return False
        for language in argos_translate.get_installed_languages():
            if language.code == self.from_code:
                return any(
                    item.to_lang.code == self.to_code
                    for item in language.translations_from
                )
        return False

    def manual_model_installation(self) -> bool:
        try:
            import argostranslate.package as argos_package
        except ImportError:
            print(
                "Argos Translate не установлен. Выполните: "
                "pip install -r requirements-optional.txt",
                flush=True,
            )
            return False

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
            argos_package.install_from_path(path)
        except Exception as error:
            print(f"Не удалось установить модель Argos: {error}", flush=True)
            return False
        print("Модель Argos установлена.", flush=True)
        return self.check_language()

    @staticmethod
    def _get_argos_translate():
        try:
            import argostranslate.translate as argos_translate
        except ImportError as error:
            raise RuntimeError(
                "Argos Translate не установлен. Для офлайн-перевода выполните: "
                "pip install -r requirements-optional.txt"
            ) from error
        return argos_translate
