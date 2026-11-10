"""Chunk-level cancellation regressions; no network, processes or models."""
import json
import unittest
from unittest.mock import patch

import httpx

from jpsub.core import translate as tr


class Chunks(httpx.SyncByteStream):
    def __init__(self, source):
        self.source = source

    def __iter__(self):
        yield from self.source()


class StreamReadReliabilityTests(unittest.TestCase):
    def providers(self):
        return [tr.OpenAICompat('http://127.0.0.1/v1', '', 'test'),
                tr.SakuraGalTransl('http://127.0.0.1'),
                tr.OllamaNative('http://127.0.0.1', 'test')]

    def translate_chunks(self, provider, chunks, control=None, *, status=200):
        transport = httpx.MockTransport(
            lambda request: httpx.Response(status, stream=Chunks(chunks)))
        client = httpx.Client
        with patch.object(tr, 'make_client', side_effect=lambda *a, **k: client(transport=transport)):
            return provider.translate('こんにちは', [], [], control)

    def test_unterminated_line_cannot_hide_deadline(self):
        for provider in self.providers():
            with self.subTest(provider=type(provider).__name__):
                now = [100.0]
                def chunks():
                    yield b'data: {'
                    now[0] = 113.0
                    yield b'x'
                    raise AssertionError('Read again after the overall deadline')
                with patch.object(tr.time, 'monotonic', side_effect=lambda: now[0]):
                    control = tr.TranslationStream(lambda _: None, lambda: False, seconds=12)
                    with self.assertRaises(tr.TranslationDeadlineExceeded):
                        self.translate_chunks(provider, chunks, control)

    def test_unterminated_line_cannot_hide_cancellation(self):
        for provider in self.providers():
            with self.subTest(provider=type(provider).__name__):
                cancelled = [False]
                def chunks():
                    yield b'data: {'
                    cancelled[0] = True
                    yield b'x'
                    raise AssertionError('Read again after cancellation')
                control = tr.TranslationStream(lambda _: None, lambda: cancelled[0])
                with self.assertRaises(tr.TranslationCancelled):
                    self.translate_chunks(provider, chunks, control)

    def test_unterminated_line_has_a_memory_limit(self):
        for provider in self.providers():
            with self.subTest(provider=type(provider).__name__):
                def chunks():
                    yield b'x' * (1024 * 1024 + 1)
                    raise AssertionError('Unbounded line accumulation')
                with self.assertRaises(tr.TranslationOutputError):
                    self.translate_chunks(provider, chunks)

    def test_valid_crlf_and_multibyte_text_can_cross_network_chunks(self):
        for provider in self.providers():
            with self.subTest(provider=type(provider).__name__):
                if isinstance(provider, tr.OllamaNative):
                    payload = {'message': {'content': '你好'}, 'done': True, 'done_reason': 'stop'}
                    text = json.dumps(payload, ensure_ascii=False) + '\r\n'
                else:
                    payload = {'choices': [{'delta': {'content': '你好'}, 'finish_reason': 'stop'}]}
                    text = ': heartbeat\r\ndata: ' + json.dumps(payload, ensure_ascii=False) + '\r\n\r\n'
                data = text.encode('utf-8')
                self.assertEqual(self.translate_chunks(provider, lambda: (data[i:i+1] for i in range(len(data)))), '你好')

    def test_http_error_does_not_wait_for_unbounded_diagnostic_body(self):
        for provider in self.providers():
            with self.subTest(provider=type(provider).__name__):
                def chunks():
                    raise AssertionError('Error status should not start a body read')
                    yield b''
                with self.assertRaises(httpx.HTTPStatusError) as caught:
                    self.translate_chunks(provider, chunks, status=503)
                self.assertIn('服务暂时不可用', tr._friendly_http_error(caught.exception))


if __name__ == '__main__':
    unittest.main()
