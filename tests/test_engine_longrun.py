"""Long-running translation regressions; mocked HTTP/processes, no model use."""
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from jpsub.core import llama_engine as le, translate as tr


def event(content='', finish=None):
    return 'data: ' + json.dumps({'choices': [
        {'delta': {'content': content}, 'finish_reason': finish}
    ]}) + '\n\n'


class FinishThenStall(httpx.SyncByteStream):
    def __iter__(self):
        yield event('你好', 'stop').encode()
        raise AssertionError('Already-finished translation waited for another network read')


class StreamCompletionTests(unittest.TestCase):
    def test_terminal_event_releases_all_openai_style_providers(self):
        providers = [tr.OpenAICompat('http://127.0.0.1/v1', '', 'test'),
                     tr.TranslateGemma('http://127.0.0.1'),
                     tr.SakuraGalTransl('http://127.0.0.1'),
                     tr.HyMT2('http://127.0.0.1')]
        for provider in providers:
            with self.subTest(provider=type(provider).__name__):
                transport = httpx.MockTransport(
                    lambda request: httpx.Response(200, stream=FinishThenStall()))
                with patch.object(tr, 'make_client', side_effect=lambda *a, **k: httpx.Client(transport=transport)):
                    self.assertEqual(provider.translate('こんにちは', [], []), '你好')

    def test_local_malformed_event_is_not_misclassified_as_provider_failure(self):
        for data in ('{broken', '[]', 'null'):
            with self.subTest(data=data):
                transport = httpx.MockTransport(lambda request: httpx.Response(200, text='data: ' + data + '\n\n'))
                with patch.object(tr, 'make_client', side_effect=lambda *a, **k: httpx.Client(transport=transport)):
                    with self.assertRaises(tr.TranslationOutputError):
                        tr.SakuraGalTransl('http://127.0.0.1').translate('こんにちは', [], [])

    def test_network_wait_cannot_exceed_remaining_initial_budget(self):
        with patch.object(tr.time, 'monotonic', return_value=100):
            control = tr.TranslationStream(None, lambda: False, seconds=10, read_timeout=15)
        with patch.object(tr.time, 'monotonic', return_value=109.5):
            timeout = tr._stream_timeout(control, 90)
        self.assertEqual(timeout.read, 0.5)
        self.assertEqual(timeout.connect, 0.5)
        self.assertEqual(timeout.write, 0.5)
        self.assertEqual(timeout.pool, 0.5)

    def test_expired_budget_never_opens_another_request(self):
        with patch.object(tr.time, 'monotonic', return_value=100):
            control = tr.TranslationStream(None, lambda: False, seconds=10)
        with patch.object(tr.time, 'monotonic', return_value=111):
            with self.assertRaises(tr.TranslationDeadlineExceeded):
                tr._stream_timeout(control, 90)

    def test_plain_callbacks_keep_legacy_timeout(self):
        timeout = tr._stream_timeout(lambda text: None, 30)
        self.assertEqual(timeout.read, 30)
        self.assertEqual(timeout.connect, 8)


class ResidentFallbackTests(unittest.TestCase):
    def setUp(self):
        self.model = Path('Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf').resolve()
        self.servers = {}
        self.patches = [patch.object(le, '_SERVERS', self.servers),
                        patch.object(le, '_FAILED', {}),
                        patch.object(le, '_LOCK', threading.Lock()),
                        patch.object(le, '_RESET_REQUESTED', threading.Event()),
                        patch.object(le, 'resolve_model', return_value=self.model),
                        patch.object(le, 'installed', return_value=True),
                        patch.object(tr.SakuraGalTransl, 'translate', return_value='你好')]
        for item in self.patches:
            item.start()
        self.addCleanup(self.cleanup_patches)

    def cleanup_patches(self):
        for item in reversed(self.patches):
            item.stop()

    def key(self, variant):
        return f'{self.model}|{variant}|2048|'

    def load(self, model, variant, **kwargs):
        if variant == 'cuda':
            raise RuntimeError('CUDA allocation failed')
        server = Mock(base_url='http://127.0.0.1/v1', mmproj=None)
        server.is_up.return_value = True
        self.servers[self.key(variant)] = server
        return server.base_url

    def test_healthy_fallback_is_reused_after_failure_cooldown(self):
        provider = tr.LlamaCppProvider({'model': str(self.model), 'variant': 'cuda'})
        with patch.object(le, 'ensure_running', side_effect=self.load) as load:
            self.assertEqual(provider.translate('こんにちは', [], []), '你好')
            # Many subsequent requests (including after the 120s cooldown) do
            # not load another GPU model alongside the working fallback.
            with patch.object(le.time, 'monotonic', return_value=10_000):
                for _ in range(100):
                    provider.translate('こんにちは', [], [])
        variants = [call.args[1] for call in load.call_args_list]
        self.assertEqual(variants.count('cuda'), 1)
        self.assertEqual(variants.count('vulkan'), 101)
        self.assertEqual(provider.actual_variant, 'vulkan')
        self.assertIn('CUDA allocation failed', provider.engine_reason)

    def test_manual_reset_retries_preferred_backend(self):
        provider = tr.LlamaCppProvider({'model': str(self.model), 'variant': 'cuda'})
        with patch.object(le, 'ensure_running', side_effect=self.load) as load:
            provider.translate('こんにちは', [], [])
            le.stop_translation()
            self.assertEqual(self.servers, {})
            provider.translate('こんにちは', [], [])
        self.assertEqual([call.args[1] for call in load.call_args_list],
                         ['cuda', 'vulkan', 'cuda', 'vulkan'])

    def test_dead_fallback_does_not_pin_provider(self):
        server = Mock()
        server.is_up.return_value = False
        self.servers[self.key('vulkan')] = server
        self.assertIsNone(le.running_variant(self.model, ['cuda', 'vulkan', 'cpu']))

    def test_other_model_and_context_are_not_reused(self):
        server = Mock()
        server.is_up.return_value = True
        self.servers[self.key('vulkan')] = server
        self.assertIsNone(le.running_variant(self.model, ['vulkan'], ctx=4096))
        self.assertIsNone(le.running_variant(Path('another.gguf'), ['vulkan']))

    def test_lookup_cancellation_releases_lock(self):
        def cancelled():
            raise tr.TranslationCancelled()
        with self.assertRaises(tr.TranslationCancelled):
            le.running_variant(self.model, ['cuda'], checkpoint=cancelled)
        self.assertTrue(le._LOCK.acquire(blocking=False))
        le._LOCK.release()


if __name__ == '__main__':
    unittest.main()
