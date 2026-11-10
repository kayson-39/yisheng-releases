"""Deterministic overload/recovery tests: no network, GPU or audio device."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore

from jpsub.core import pipeline


class TestConfig:
    def get(self, key, default=None):
        return default

    def service(self, name):
        return {}


class RealtimePipelineTests(unittest.TestCase):
    def setUp(self):
        self.job = pipeline.Pipeline(TestConfig())
        self.failed = []
        self.errors = []
        self.outputs = []
        self.job.sig_translate_failed.connect(
            lambda *args: self.failed.append(args), QtCore.Qt.DirectConnection)
        self.job.sig_error.connect(self.errors.append, QtCore.Qt.DirectConnection)
        self.job.sig_final_zh.connect(
            lambda *args: self.outputs.append(args), QtCore.Qt.DirectConnection)

    def tearDown(self):
        self.job._stop.set()
        self.job._shutdown_pools()
        self.job._release_resources()

    def defer_workers(self):
        submitted = []
        self.job._submit = lambda pool, fn, *args: submitted.append((fn, args)) or object()
        return submitted

    def fake_translator(self, translate):
        return SimpleNamespace(translate=translate, last_provider="", last_engine="",
                               last_engine_reason="")

    def test_translation_overload_bounds_real_work_and_marks_old_japanese_retryable(self):
        submitted = self.defer_workers()
        for number in range(1000):
            self.assertTrue(self.job.enqueue_translation(str(number), "日本語の原文"))
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(self.job._pending_final), pipeline.MAX_PENDING_FINALS)
        self.assertEqual(len(self.job._translation_jobs), pipeline.MAX_PENDING_FINALS)
        self.assertEqual(list(self.job._translation_jobs), [str(n) for n in range(992, 1000)])
        self.assertEqual(len(self.failed), 992)
        self.assertTrue(all("原文已保留" in message for _, message in self.failed))
        translated = []
        self.job._run_translate_final = lambda sid, text, context: translated.append(sid)
        self.job._drain_translations()
        self.assertEqual(translated, [str(n) for n in range(992, 1000)])
        self.assertFalse(self.job._pending_final)
        self.assertFalse(self.job._translation_worker_running)

    def test_manual_retry_does_not_evict_live_or_other_retry_work(self):
        self.defer_workers()
        for number in range(pipeline.MAX_PENDING_FINALS):
            self.assertTrue(self.job.enqueue_translation(str(number), "日本語"))
        pending = dict(self.job._pending_final)
        self.assertFalse(self.job.enqueue_translation("extra", "日本語", retry=True))
        self.assertFalse(self.job.enqueue_translation("0", "日本語", retry=True))
        self.assertEqual(self.job._pending_final, pending)
        self.assertFalse(self.failed)

    def test_expired_final_does_not_run_model_and_manual_retry_gets_fresh_budget(self):
        self.defer_workers()
        self.job.enqueue_translation("old", "日本語")
        self.job._pending_final["old"] = time.monotonic() - 100
        with patch.object(pipeline, "Translator") as translator:
            self.job._drain_translations()
            translator.assert_not_called()
        self.assertEqual(self.failed[0][0], "old")
        self.assertIn("排队", self.failed[0][1])
        self.assertTrue(self.job.enqueue_translation("old", "日本語", retry=True))
        self.assertLess(time.monotonic() - self.job._pending_final["old"], 1)
        with patch.object(pipeline, "Translator", return_value=self.fake_translator(
                lambda *args, **kw: "完成补译")):
            self.job._drain_translations()
        self.assertNotIn("old", self.job._failed_translations)
        self.assertIn(("old", "完成补译"), self.outputs)

    def test_retries_share_one_deadline_including_queue_time(self):
        clock = [100.0]
        deadlines = []
        self.job._pending_final["s"] = 96.0
        def translate(*args, on_delta=None):
            deadlines.append(on_delta.deadline)
            if len(deadlines) == 1:
                clock[0] += 3
                raise pipeline.TranslationOutputError("bad output")
            return "正确译文"
        with patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), patch.object(
                pipeline, "Translator", return_value=self.fake_translator(translate)):
            self.job._translate_final_locked("s", "日本語", [])
        self.assertEqual(deadlines, [108.0, 108.0])
        self.assertFalse(self.failed)
        self.assertIn(("s", "正确译文"), self.outputs)

    def test_timed_out_stream_cannot_emit_or_restart_full_retry_budget(self):
        clock = [100.0]
        calls = []
        def translate(*args, on_delta=None):
            calls.append(True)
            clock[0] += 13
            on_delta("迟到的译文")
            return "迟到的译文"
        with patch.object(pipeline.time, "monotonic", side_effect=lambda: clock[0]), patch.object(
                pipeline, "Translator", return_value=self.fake_translator(translate)):
            self.job._translate_final_locked("s", "日本語", [])
        self.assertEqual(calls, [True])
        self.assertEqual(self.outputs, [])
        self.assertIn("实时等待上限", self.failed[0][1])

    def test_unexpected_worker_exception_does_not_strand_later_jobs(self):
        self.defer_workers()
        self.job.enqueue_translation("bad", "日本語")
        self.job.enqueue_translation("good", "日本語")
        seen = []
        def run(sid, text, context):
            seen.append(sid)
            if sid == "bad":
                raise ValueError("test failure")
        self.job._run_translate_final = run
        self.job._drain_translations()
        self.assertEqual(seen, ["bad", "good"])
        self.assertFalse(self.job._translation_worker_running)
        self.assertFalse(self.job._pending_final)

    def test_asr_audio_work_is_bounded_before_executor_and_drafts_yield(self):
        submitted = self.defer_workers()
        for number in range(1000):
            self.job._submit_final_audio(np.zeros(4000, dtype=np.float32), number, "")
        self.assertEqual(len(submitted), 1)
        self.assertEqual(len(self.job._asr_jobs), pipeline.MAX_PENDING_FINALS)
        self.assertEqual(len(self.errors), 992)
        self.assertTrue(all("未完成识别" in error for error in self.errors))
        self.job._draft_mode = "rolling"
        self.job._draft_engine = SimpleNamespace(close=lambda: None)
        self.job._utterance = [np.zeros(16000, dtype=np.float32)]
        self.job._maybe_rolling_draft(time.monotonic())
        self.assertEqual(len(submitted), 1)
        processed = []
        self.job._run_final = lambda seg, samples: processed.append(seg.t0)
        self.job._drain_final_audio()
        self.assertEqual(processed, list(range(992, 1000)))
        self.assertFalse(self.job._asr_worker_running)

    def test_audio_overload_keeps_newest_and_resets_vad_at_real_sample_position(self):
        self.job._queue.maxsize = 3
        for number in range(4):
            self.job._on_audio(np.full(1280, number, dtype=np.float32), .5)
        self.assertEqual(self.job._queue.qsize(), 1)
        resets, observed = [], []
        self.job._vad = SimpleNamespace(reset=lambda: resets.append(True))
        self.job._setup = lambda: None
        def process(samples, rms):
            observed.append((samples[0], self.job._fed, self.job._vad_origin))
            self.job._stop.set()
        self.job._process_chunk = process
        self.job.run()
        self.assertEqual(resets, [True])
        self.assertEqual(observed, [(3.0, 3840, 3840)])
        self.assertEqual(len(self.errors), 1)

    def test_asr_unexpected_item_error_does_not_strand_worker_or_later_audio(self):
        submitted = self.defer_workers()
        audio = np.zeros(4000, dtype=np.float32)
        self.job._submit_final_audio(audio, 0, "")
        self.job._submit_final_audio(audio, 1, "")
        seen = []
        def run(seg, samples):
            seen.append(seg.t0)
            if seg.t0 == 0:
                raise ValueError("test-only post-decode failure")
        self.job._run_final = run
        self.job._drain_final_audio()
        self.assertEqual(seen, [0, 1])
        self.assertFalse(self.job._asr_jobs)
        self.assertFalse(self.job._asr_worker_running)
        self.assertEqual(len(self.errors), 1)
        self.assertIn("test-only post-decode failure", self.errors[0])
        self.job._submit_final_audio(audio, 2, "")
        self.assertEqual(len(submitted), 2)
        self.job._drain_final_audio()
        self.assertEqual(seen, [0, 1, 2])
        self.assertFalse(self.job._asr_worker_running)

    def test_invalid_translation_age_limit_uses_default(self):
        for value in (None, "bad", [], {}, True, False, float("nan"),
                      float("inf"), float("-inf")):
            with self.subTest(value=value), patch.object(self.job.cfg, "get", return_value=value):
                self.assertEqual(self.job._translation_age_limit(), pipeline.MAX_TRANSLATION_AGE_S)

    def test_valid_translation_age_limit_is_clamped(self):
        for value, expected in (("15.5", 15.5), (10, 10), (-1, 2), (120, 60)):
            with self.subTest(value=value), patch.object(self.job.cfg, "get", return_value=value):
                self.assertEqual(self.job._translation_age_limit(), expected)

    def test_stop_releases_queued_payloads_and_no_new_work_is_accepted(self):
        self.defer_workers()
        self.job.enqueue_translation("s", "日本語")
        self.job._submit_final_audio(np.zeros(4000, dtype=np.float32), 0, "")
        self.job._stop.set()
        self.job._drain_translations()
        self.job._drain_final_audio()
        self.assertFalse(self.job.enqueue_translation("new", "日本語"))
        self.assertFalse(self.job._translation_jobs)
        self.assertFalse(self.job._pending_final)
        self.assertFalse(self.job._asr_jobs)

    def test_worker_races_with_enqueue_do_not_lose_new_jobs(self):
        entered, release = threading.Event(), threading.Event()
        processed = []
        def run(sid, text, context):
            processed.append(sid)
            if sid == "first":
                entered.set()
                self.assertTrue(release.wait(2))
        self.job._run_translate_final = run
        self.assertTrue(self.job.enqueue_translation("first", "日本語"))
        self.assertTrue(entered.wait(2))
        self.assertTrue(self.job.enqueue_translation("second", "日本語"))
        release.set()
        self.job._trans_pool.shutdown(wait=True)
        self.assertEqual(processed, ["first", "second"])
        self.assertFalse(self.job._pending_final)


if __name__ == "__main__":
    unittest.main()
