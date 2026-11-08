import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from core.config import AppConfig
from modules.translation.google import GoogleTranslateError, GoogleTranslateProvider
from modules.translation.translator import Translator


class GoogleProviderTests(unittest.TestCase):
    def provider(self, handler):
        provider = GoogleTranslateProvider()
        provider._client.close()
        provider._client = httpx.Client(transport=httpx.MockTransport(handler))
        self.addCleanup(provider.close)
        return provider

    def test_primary_segments_are_joined_and_source_sent_as_post_form(self):
        def handler(request):
            self.assertEqual(request.method, 'POST')
            self.assertEqual(request.url.params['client'], 'gtx')
            self.assertNotIn('q', request.url.params)
            self.assertIn(b'q=Hello', request.content)
            return httpx.Response(200, json=[[['Привет. ', 'Hello.'], ['Мир!', 'World!']], None, 'en'])
        provider = self.provider(handler)
        self.assertEqual(provider.translate('Hello. World!'), 'Привет. Мир!')

    def test_rate_limited_primary_uses_reserve_and_remembers_it(self):
        hosts = []
        def handler(request):
            hosts.append(request.url.host)
            if request.url.host == 'translate.googleapis.com':
                return httpx.Response(429, text='limited')
            self.assertEqual(request.url.params['client'], 'dict-chrome-ex')
            return httpx.Response(200, json=['Перевод'])
        provider = self.provider(handler)
        self.assertEqual(provider.translate('Hello'), 'Перевод')
        self.assertEqual(provider.translate('World'), 'Перевод')
        self.assertEqual(hosts, ['translate.googleapis.com', 'clients5.google.com', 'clients5.google.com'])

    def test_malformed_or_empty_primary_response_uses_reserve(self):
        for payload in ({}, [], [None], [[None]], [''], [[[None]]]):
            with self.subTest(payload=payload):
                def handler(request):
                    value = payload if request.url.host == 'translate.googleapis.com' else ['Привет']
                    return httpx.Response(200, json=value)
                self.assertEqual(self.provider(handler).translate('Hello'), 'Привет')

    def test_html_response_is_not_shown_as_translation(self):
        provider = self.provider(lambda request: httpx.Response(200, text='<html>Consent</html>'))
        with self.assertRaises(RuntimeError):
            provider.translate('Hello')

    def test_both_endpoints_blocked_preserve_status_for_frame_circuit_breaker(self):
        provider = self.provider(lambda request: httpx.Response(429))
        with self.assertRaises(RuntimeError) as error:
            provider.translate('Hello')
        self.assertIsInstance(error.exception.__cause__, httpx.HTTPStatusError)
        self.assertIn('429', str(error.exception))

    def test_failed_reserve_can_return_to_primary(self):
        def handler(request):
            return (httpx.Response(503) if request.url.host == 'clients5.google.com'
                    else httpx.Response(200, json=[[['Привет', 'Hello']]]))
        provider = self.provider(handler)
        provider._preferred = 1
        self.assertEqual(provider.translate('Hello'), 'Привет')
        self.assertEqual(provider._preferred, 0)

    def test_chunk_limit_and_order(self):
        requests = []
        from urllib.parse import parse_qs
        def handler(request):
            text = parse_qs(request.content.decode())['q'][0]
            requests.append(text)
            self.assertLessEqual(len(text), GoogleTranslateProvider.MAX_CHARS)
            return httpx.Response(200, json=[text.upper()])
        text = 'a'*4600+'b'*4600
        result = self.provider(handler).translate(text)
        self.assertEqual(result.replace('\n', ''), text.upper())
        self.assertEqual(len(requests), 3)

    def test_empty_text_does_not_request(self):
        handler = Mock()
        self.assertEqual(self.provider(handler).translate(''), '')
        handler.assert_not_called()


class GoogleIntegrationTests(unittest.TestCase):
    def test_unusable_google_responses_stop_remaining_frame_requests(self):
        translator = self.translator(provider='google')
        translator._google_provider = Mock()
        translator._google_provider.translate.side_effect = GoogleTranslateError('invalid web response')
        result = translator.translate_many([f'Text {i}' for i in range(20)])
        self.assertEqual(len(result),20)
        self.assertTrue(all(isinstance(value,Exception) for value in result.values()))
        self.assertLessEqual(translator._google_provider.translate.call_count,3)

    def translator(self, **kwargs):
        translator = Translator(**kwargs)
        self.addCleanup(translator.close)
        return translator

    def test_google_is_cached_and_deduplicated(self):
        translator = self.translator(provider='google')
        translator._google_provider = Mock()
        translator._google_provider.translate.return_value = 'Привет'
        self.assertEqual(translator.translate_many(['Hello', 'Hello']), {'Hello': 'Привет'})
        self.assertEqual(translator.translate_many(['Hello']), {'Hello': 'Привет'})
        translator._google_provider.translate.assert_called_once_with('Hello')

    def test_google_can_be_primary_or_fallback(self):
        for primary, fallback in [('google', 'mymemory'), ('mymemory', 'google')]:
            with self.subTest(primary=primary):
                translator = self.translator(provider=primary, fallback_provider=fallback, hybrid_enabled=True)
                translator._google_provider = Mock()
                translator._mymemory_provider = Mock()
                getattr(translator, '_'+primary+'_provider').translate.side_effect = httpx.ConnectError('offline')
                getattr(translator, '_'+fallback+'_provider').translate.return_value = 'Привет'
                self.assertEqual(translator.translate_many(['Hello']), {'Hello': 'Привет'})

    def test_configuration_accepts_google(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AppConfig(Path(directory))
            config.path = Path(directory)/'settings.json'
            config.save({'provider': 'google', 'fallback_provider': 'mymemory'})
            settings = config.load()
            self.assertEqual(settings['provider'], 'google')
            self.assertEqual(settings['fallback_provider'], 'mymemory')
