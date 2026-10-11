from typing import Protocol


class Translator(Protocol):
    """Минимальный интерфейс любого переводчика TranScr."""

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ru",
    ) -> str:
        ...
