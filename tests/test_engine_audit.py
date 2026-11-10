"""Offline regressions: mocked model processes and HTTP streams, no GPU/cloud use."""
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from jpsub.core import llama_engine as le, translate as tr, model_catalog
from jpsub.paths import MODELS


def event(content='', finish=None):
    return 'data: ' + json.dumps({'choices': [{'delta': {'content': content}, 'finish_reason': finish}]})


class Streams(unittest.TestCase):
    def run_stream(self, provider, lines, callback=None):
        transport = httpx.MockTransport(lambda req: httpx.Response(200, text='\n'.join(lines)))
        with patch.object(tr, 'make_client', side_effect=lambda *a, **k: httpx.Client(transport=transport)):
            return provider.translate('こんにちは', [], [], callback)

    def openai(self):
        return tr.OpenAICompat('http://127.0.0.1/v1', '', 'test')

    def test_openai_rejects_premature_eof(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(self.openai(), [event('半句')])

    def test_openai_rejects_length_limit(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(self.openai(), [event('半句'), event(finish='length'), 'data: [DONE]'])

    def test_openai_does_not_swallow_server_error(self):
        with self.assertRaisesRegex(RuntimeError, 'overloaded'):
            self.run_stream(self.openai(), [event('半句'), 'data: {"error":"overloaded"}', 'data: [DONE]'])

    def test_openai_rejects_malformed_event(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(self.openai(), [event('半句'), 'data: {broken', 'data: [DONE]'])

    def test_callback_failure_is_not_ignored(self):
        def callback(piece):
            raise ValueError('consumer failed')
        with self.assertRaisesRegex(ValueError, 'consumer failed'):
            self.run_stream(self.openai(), [event('你好', 'stop')], callback)

    def test_valid_stop_without_done_is_compatible(self):
        self.assertEqual(self.run_stream(self.openai(), [event('你好'), event(finish='stop')]), '你好')

    def test_done_and_usage_only_event_are_compatible(self):
        lines = [': heartbeat', event('你好'), 'data: {"choices": [], "usage": {}}', 'data: [DONE]']
        self.assertEqual(self.run_stream(self.openai(), lines), '你好')

    def test_gemma_rejects_premature_eof(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(tr.TranslateGemma('http://127.0.0.1'), [event('半句')])

    def test_gemma_rejects_length_limit(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(tr.TranslateGemma('http://127.0.0.1'), [event('半句', 'length')])

    def test_sakura_valid_response_keeps_text(self):
        self.assertEqual(self.run_stream(tr.SakuraGalTransl('http://127.0.0.1'), [event('你好！', 'stop')]), '你好！')

    def test_ollama_rejects_premature_eof(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(tr.OllamaNative('http://127.0.0.1', 'test'), ['{"message":{"content":"半句"}}'])

    def test_ollama_rejects_length_limit(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(tr.OllamaNative('http://127.0.0.1', 'test'), ['{"message":{"content":"半句"},"done":true,"done_reason":"length"}'])

    def test_ollama_rejects_malformed_event(self):
        with self.assertRaises(tr.TranslationOutputError):
            self.run_stream(tr.OllamaNative('http://127.0.0.1', 'test'), ['{"message":{"content":"半句"}}', '{broken', '{"done":true}'])

    def test_ollama_valid_response_keeps_text(self):
        self.assertEqual(self.run_stream(tr.OllamaNative('http://127.0.0.1', 'test'), ['{"message":{"content":"你好"},"done":true,"done_reason":"stop"}']), '你好')


class EngineLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.model = self.folder / 'test.gguf'
        self.model.write_bytes(b'GGUF')
        self.exe = self.folder / 'llama-server.exe'
        self.exe.write_bytes(b'not executed')
        self.patches = [patch.object(le, 'ENGINE_DIR', self.folder), patch.object(le, '_SERVERS', {}),
                        patch.object(le, '_FAILED', {}), patch.object(le, '_LOCK', threading.Lock()),
                        patch.object(le, 'SERVER_LOG', self.folder / 'server.log', create=True),
                        patch.object(le, '_join_job'), patch.object(le, 'stale_servers'),
                        patch.object(le, 'server_exe', return_value=self.exe)]
        for item in self.patches: item.start()

    def tearDown(self):
        le.stop_all()
        for item in reversed(self.patches): item.stop()
        self.tmp.cleanup()

    def test_cold_load_cancel_kills_child_and_allows_retry(self):
        cancelled = threading.Event()
        child = Mock(pid=12345)
        child.poll.return_value = None
        def spawn(*args, **kwargs):
            cancelled.set()
            return child
        callback = tr.TranslationStream(None, cancelled.is_set)
        with patch.object(le.subprocess, 'Popen', side_effect=spawn), self.assertRaises(tr.TranslationCancelled):
            le.ensure_running(self.model, 'cpu', checkpoint=callback.checkpoint)
        child.terminate.assert_called_once()
        child.wait.assert_called_once()
        self.assertEqual(le._FAILED, {})
        self.assertEqual(le._SERVERS, {})
        self.assertTrue(le._LOCK.acquire(blocking=False))
        le._LOCK.release()

    def test_lock_wait_can_cancel(self):
        cancelled = threading.Event()
        callback = tr.TranslationStream(None, cancelled.is_set)
        le._LOCK.acquire()
        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        started = time.monotonic()
        try:
            with self.assertRaises(tr.TranslationCancelled):
                le.ensure_running(self.model, checkpoint=callback.checkpoint)
        finally:
            timer.join()
            le._LOCK.release()
        self.assertLess(time.monotonic() - started, 0.8)

    def test_different_context_is_not_reused(self):
        created = []
        def make(*args, **kwargs):
            server = Mock(base_url=f'url{len(created)}', mmproj=kwargs.get('mmproj'), _proc=None)
            server.is_up.return_value = True
            created.append(server)
            return server
        with patch.object(le, 'LlamaServer', side_effect=make):
            first = le.ensure_running(self.model, ctx=1024)
            second = le.ensure_running(self.model, ctx=2048)
        self.assertNotEqual(first, second)
        self.assertEqual(len(created), 2)

    def test_failure_diagnostics_reads_only_tail(self):
        (self.folder / 'server.log').write_bytes(b'x' * 100_000 + b'failed to allocate')
        with patch.object(Path, 'read_bytes', side_effect=AssertionError('unbounded log read')):
            self.assertIn('显卡', le._diagnose_start_failure(RuntimeError('failed')))

    def test_manual_reset_clears_failure_cooldown(self):
        le._FAILED['previous'] = (time.monotonic(), 'out of memory')
        le.stop_translation()
        self.assertEqual(le._FAILED, {})

    def test_manual_reset_interrupts_cold_load(self):
        started, done = threading.Event(), threading.Event()
        errors = []
        server = Mock(mmproj=None)
        def start(**kwargs):
            started.set()
            until = time.monotonic() + 1.5
            while time.monotonic() < until:
                if kwargs.get('checkpoint'): kwargs['checkpoint']()
                time.sleep(0.01)
        server.start.side_effect = start
        def load():
            try: le.ensure_running(self.model)
            except Exception as exc: errors.append(exc)
            finally: done.set()
        with patch.object(le, 'LlamaServer', return_value=server):
            worker = threading.Thread(target=load)
            worker.start()
            self.assertTrue(started.wait(1))
            before = time.monotonic()
            le.stop_translation()
            elapsed = time.monotonic() - before
            worker.join(2)
        self.assertLess(elapsed, 0.8)
        self.assertTrue(done.is_set())
        self.assertTrue(any(isinstance(exc, tr.TranslationCancelled) for exc in errors))
        self.assertEqual(le._FAILED, {})

    def test_invalid_runtime_profile_uses_defaults(self):
        with patch.object(le, '_runtime_profile', return_value={'context': None, 'threads': 'oops'}):
            server = le.LlamaServer(self.model, ctx=2048, threads=4)
        self.assertEqual(server.ctx, 2048)
        self.assertEqual(server.threads, 4)

    def test_non_object_profile_file_is_ignored(self):
        (self.folder / 'runtime_profiles.json').write_text('[]', encoding='utf-8')
        self.assertEqual(le._runtime_profile(self.model), {})

    def test_start_only_writes_log_in_user_directory(self):
        engine = self.folder / 'readonly-install'
        userdata = self.folder / 'userdata'
        child = Mock(pid=12345)
        child.poll.return_value = None
        server = le.LlamaServer(self.model)
        with patch.object(le, 'ENGINE_DIR', engine), patch.object(le, 'SERVER_LOG', userdata / 'server.log', create=True), patch.object(le.subprocess, 'Popen', return_value=child), patch.object(server, 'is_up', return_value=True):
            server.start()
            server.stop()
        self.assertTrue((userdata / 'server.log').is_file())
        self.assertFalse(engine.exists())


class DownloadTests(unittest.TestCase):
    def test_corrupt_existing_model_is_repaired(self):
        good = b'GGUF' + b'a' * (1024 * 1024)
        bad = b'GGUF' + b'b' * (1024 * 1024)
        item = {'name': 'test.gguf', 'repo': 'test/repo', 'sha256': hashlib.sha256(good).hexdigest()}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            target = folder / item['name']
            target.write_bytes(bad)
            client = httpx.Client(transport=httpx.MockTransport(lambda req: httpx.Response(200, content=good)))
            with client, patch.object(le, 'MODELS_LLM', folder), patch.object(le.httpx, 'stream', side_effect=client.stream):
                self.assertEqual(le.download_model(item), target)
            self.assertEqual(target.read_bytes(), good)

    def test_catalog_uses_same_model_directory_as_runtime(self):
        self.assertEqual(model_catalog.MODELS_DIR, MODELS)

    def test_zero_length_asr_download_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'model.int8.onnx').touch()
            (folder / 'tokens.txt').touch()
            with patch.object(model_catalog, 'target_dir', return_value=folder):
                self.assertFalse(model_catalog.is_complete('parakeet-ja'))


class FallbackTests(unittest.TestCase):
    def test_slow_fallback_wait_can_cancel(self):
        cfg = SimpleNamespace(get=lambda name, default=None: default, service=lambda name: {})
        translator = tr.Translator(cfg)
        cancelled, release = threading.Event(), threading.Event()
        callback = tr.TranslationStream(None, cancelled.is_set)
        def slow(*args):
            release.wait(1)
            return '你好'
        timer = threading.Timer(0.15, cancelled.set)
        timer.start()
        started = time.monotonic()
        try:
            with patch.object(translator, '_order_fallbacks', return_value=['test']), patch.object(tr, 'make_provider', return_value=SimpleNamespace(translate=slow)):
                with self.assertRaises(tr.TranslationCancelled):
                    translator._race_keyless('こんにちは', [], [], callback)
        finally:
            release.set()
            timer.join()
        self.assertLess(time.monotonic() - started, 0.8)


if __name__ == '__main__':
    unittest.main()
