from googletrans import Translator as GoogleTranslatorClient


class GoogleTranslator:
    """Google Translate через неофициальный googletrans web-клиент."""

    def __init__(self) -> None:
        self._client = GoogleTranslatorClient()

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ru",
    ) -> str:
        text = text.strip()
        if not text:
            return ""

        result = self._client.translate(
            text,
            src=source_lang,
            dest=target_lang,
        )
        return result.text.strip()
