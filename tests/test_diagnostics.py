"""Numeric-only, bounded diagnostics; no models, devices or cloud calls."""
import json
import logging
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from jpsub.core import pipeline


class TestConfig:
    def get(self, key, default=None):
        return default

    def service(self, name):
        return {}


class PerformanceDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.job = pipeline.Pipeline(TestConfig())

    def tearDown(self):
        self.job._stop.set()
        self.job._shutdown_pools()
        self.job._release_resources()

    def test_throttled_numeric_snapshot_without_transcript_or_identifiers(self):
        # At this origin (base + 30) - base rounds below 30. The exact
        # scheduled instant is due nevertheless, regardless of host uptime.
        base = self.job._performance_started = self.job._last_performance_report = 2.001
        self.job._on_audio(np.zeros(1280, dtype=np.float32), 0.0)
        self.job._on_audio(np.zeros(640, dtype=np.float32), 0.0)
        self.job._asr_jobs['private-source-id'] = ('private source',)
        self.job._final_busy = 1
        rows = []
        with patch.object(pipeline, '_write_performance', side_effect=rows.append):
            self.job._report_performance(base + 29.9, 2, 4.5, 3)
            self.assertFalse(rows)
            self.job._report_performance(base + 30, 2, 4.5, 3)
            self.job._report_performance(base + 30.1, 2, 4.5, 3)
            self.assertEqual(len(rows), 1)
            self.job._report_performance(base + 60, 1, 0.1, 3)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['audio_queue'], 2)
        self.assertEqual(rows[0]['audio_buffered_s'], 0.12)
        self.assertEqual(rows[0]['asr_queued'], 1)
        self.assertEqual(rows[0]['asr_running'], 1)
        self.assertEqual(rows[0]['translation_pending'], 2)
        self.assertEqual(rows[0]['translation_oldest_s'], 4.5)
        self.assertEqual(rows[0]['translation_failed'], 3)
        self.assertTrue(all(type(value) in (int, float) for row in rows for value in row.values()))
        self.assertNotIn('private', json.dumps(rows))

    def test_busy_queue_lock_skips_sample_without_blocking(self):
        with self.job._queue.mutex:
            with patch.object(pipeline, '_write_performance') as writer:
                self.job._report_performance(self.job._performance_started + 30, 0, 0, 0)
        writer.assert_not_called()
        self.assertTrue(self.job._performance_report_lock.acquire(blocking=False))
        self.job._performance_report_lock.release()

    def test_busy_asr_lock_skips_sample_without_blocking(self):
        with self.job._asr_state_lock:
            with patch.object(pipeline, '_write_performance') as writer:
                self.job._report_performance(self.job._performance_started + 30, 0, 0, 0)
        writer.assert_not_called()

    def test_logging_error_cannot_break_queue_reporting(self):
        rows = []
        from PySide6 import QtCore
        self.job.sig_translate_queue.connect(rows.append, QtCore.Qt.DirectConnection)
        self.job._last_performance_report -= 31
        with patch.object(pipeline, '_write_performance', side_effect=OSError('disk unavailable')):
            self.job._report_translation_queue(True)
        self.assertEqual(rows, [{'pending': 0, 'oldest_s': 0, 'failed': 0,
                                 'audio_pending': 0, 'asr_pending': 0}])

    def test_ui_queue_state_reports_upstream_audio_and_running_asr(self):
        rows = []
        from PySide6 import QtCore
        self.job.sig_translate_queue.connect(rows.append, QtCore.Qt.DirectConnection)
        self.job._on_audio(np.zeros(1280, dtype=np.float32), 0.0)
        self.job._asr_jobs['waiting'] = ('source',)
        self.job._final_busy = 1
        self.job._report_translation_queue(True)
        self.assertEqual(rows[-1]['audio_pending'], 1)
        self.assertEqual(rows[-1]['asr_pending'], 2)
        self.job._queue.get_nowait()
        self.job._asr_jobs.clear()
        self.job._report_translation_queue(True)
        self.assertEqual(rows[-1]['audio_pending'], 0)
        self.assertEqual(rows[-1]['asr_pending'], 1)

    def test_writer_created_once_across_multiple_pipeline_sessions(self):
        with patch.object(pipeline, '_performance_writer', None):
            with patch.object(pipeline, '_PerformanceWriter') as constructor:
                pipeline._write_performance({'elapsed_s': 30})
                pipeline._write_performance({'elapsed_s': 60})
        constructor.assert_called_once_with(pipeline.USER_DATA / 'performance.log')
        self.assertEqual(constructor.return_value.submit.call_count, 2)

    def test_bounded_queue_drops_samples_without_waiting(self):
        writer = pipeline._PerformanceWriter.__new__(pipeline._PerformanceWriter)
        writer.queue = queue.Queue(maxsize=2)
        for value in range(100):
            writer.submit({'elapsed_s': value})
        self.assertEqual(writer.queue.qsize(), 2)
        self.assertEqual(json.loads(writer.queue.get_nowait().msg), {'elapsed_s': 0})

    def test_handler_is_lazy_rotating_and_suppresses_disk_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'child' / 'performance.log'
            handler = pipeline._PerformanceFileHandler(
                path, maxBytes=512*1024, backupCount=2, encoding='utf-8', delay=True)
            self.addCleanup(handler.close)
            self.assertIsNone(handler.stream)
            self.assertFalse(path.exists())
            self.assertEqual(handler.maxBytes, 512*1024)
            self.assertEqual(handler.backupCount, 2)
            record = logging.LogRecord('test', logging.INFO, '', 0, '{"elapsed_s":30}', (), None)
            with patch.object(handler, '_open', side_effect=OSError('read-only')):
                handler.handle(record)
            handler.handle(record)
            handler.close()
            self.assertEqual(json.loads(path.read_text('utf-8')), {'elapsed_s': 30})

    def test_slow_disk_is_not_on_caller_thread(self):
        entered, release = threading.Event(), threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            writer = pipeline._PerformanceWriter(Path(directory) / 'performance.log')
            try:
                def slow(record):
                    entered.set()
                    release.wait(2)
                with patch.object(writer.handler, 'emit', side_effect=slow):
                    writer.submit({'elapsed_s': 30})
                    self.assertTrue(entered.wait(1))
                    started = time.monotonic()
                    for value in range(100):
                        writer.submit({'elapsed_s': value})
                    self.assertLess(time.monotonic() - started, 0.5)
                    self.assertLessEqual(writer.queue.qsize(), 16)
                    release.set()
                    writer.queue.join()
            finally:
                release.set()
                writer.queue.join()
                writer.listener.stop()
                writer.handler.close()


if __name__ == '__main__':
    unittest.main()
