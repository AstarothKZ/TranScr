import asyncio


class GoogleTranslator:
    """Google Translate через асинхронный googletrans, с синхронным интерфейсом для worker."""

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "ru",
    ) -> str:
        text = text.strip()
        if not text:
            return ""

        # googletrans 4.0.2 uses async methods. TranslationWorker has a regular
        # synchronous interface and runs in its own thread, so bridge the API here.
        return asyncio.run(
            self._translate_async(
                text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )

    @staticmethod
    async def _translate_async(
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        # Create and close the async client inside the event loop that uses it.
        # This avoids reusing an HTTP client across separate asyncio.run() loops.
        from googletrans import Translator as GoogleTranslatorClient

        async with GoogleTranslatorClient() as client:
            result = await client.translate(
                text,
                src=source_lang,
                dest=target_lang,
            )
        return result.text.strip()
