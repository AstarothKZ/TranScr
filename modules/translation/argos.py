class ArgosTranslator:
    """Локальный переводчик Argos Translate; языковые модели ставятся отдельно."""

    def __init__(self) -> None:
        try:
            import argostranslate.translate as argos_translate
        except ImportError as error:
            raise RuntimeError(
                "Argos Translate не установлен. Установите зависимости из requirements.txt."
            ) from error
        self._argos_translate = argos_translate

    def validate_language_pair(self, source_lang: str, target_lang: str) -> None:
        """Проверяет наличие прямой установленной модели до первого OCR."""
        installed_languages = self._argos_translate.get_installed_languages()
        source = next(
            (language for language in installed_languages if language.code == source_lang),
            None,
        )
        target = next(
            (language for language in installed_languages if language.code == target_lang),
            None,
        )
        if source is None or target is None or source.get_translation(target) is None:
            raise RuntimeError(
                f"Не найдена установленная модель Argos для пары "
                f"'{source_lang} -> {target_lang}'. Установите эту языковую пару."
            )

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ru",
    ) -> str:
        text = text.strip()
        if not text:
            return ""
        if source_lang == "auto":
            raise ValueError(
                "Argos Translate требует исходный язык, например 'en'. "
                "Автоопределение языка для этого модуля пока не поддерживается."
            )

        self.validate_language_pair(source_lang, target_lang)
        installed_languages = self._argos_translate.get_installed_languages()
        source = next(language for language in installed_languages if language.code == source_lang)
        target = next(language for language in installed_languages if language.code == target_lang)
        translation = source.get_translation(target)
        return translation.translate(text).strip()
