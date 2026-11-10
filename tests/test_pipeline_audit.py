"""Device and worker lifecycle regressions; no microphone, network or GPU use."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore

from jpsub.core import audio, pipeline


class TestConfig:
    def get(self, key, default=None):
        return default

    def service(self, name):
        return {}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.job = pipeline.Pipeline(TestConfig())
        self.partials = []
        self.job.sig_partial.connect(self.partials.append, QtCore.Qt.DirectConnection)

    def tearDown(self):
        self.job._stop.set()
        self.job._shutdown_pools()
        self.job._release_resources()

    def decode_then(self, change_state):
        self.job._active = True
        self.job._draft_busy = True
        self.job._draft_started = time.monotonic()
        def transcribe(samples):
            change_state()
            return SimpleNamespace(text="古い字幕です")
        self.job._draft_engine = SimpleNamespace(transcribe=transcribe, close=lambda: None)
        self.job._run_draft(self.job._live["gen"], np.zeros(16000, dtype=np.float32))

    def test_inflight_draft_cannot_emit_after_stop(self):
        self.decode_then(self.job._stop.set)
        self.assertEqual(self.partials, [])
        self.assertFalse(self.job._draft_busy)

    def test_inflight_draft_cannot_emit_while_paused(self):
        self.decode_then(lambda: self.job.pause(True))
        self.assertEqual(self.partials, [])

    def test_quick_pause_resume_invalidates_inflight_draft(self):
        def toggle():
            self.job.pause(True)
            self.job.pause(False)
        self.decode_then(toggle)
        self.assertEqual(self.partials, [])

    def test_quick_pause_resume_invalidates_inflight_live_translation(self):
        updates = []
        self.job._active = True
        self.job.sig_live_zh.connect(lambda *args: updates.append(args), QtCore.Qt.DirectConnection)
        def translate(text, context, provider=None, on_delta=None):
            self.job.pause(True)
            self.job.pause(False)
            on_delta("旧译文")
            return "旧译文"
        with patch.object(pipeline, "Translator", return_value=SimpleNamespace(translate=translate)):
            self.job._run_translate_live(self.job._live['gen'], 'live-test', '古い字幕です')
        self.assertEqual(updates, [])
        self.assertFalse(self.job._live_task_inflight)

    def test_paused_capture_is_not_queued_and_resume_drops_old_audio(self):
        samples = np.ones(512, dtype=np.float32)
        self.job._on_audio(samples, 0.5)
        self.job.pause(True)
        self.job._on_audio(samples, 0.5)
        self.assertEqual(self.job._queue.qsize(), 1)
        self.job.pause(False)
        self.assertTrue(self.job._queue.empty())
        self.assertTrue(self.job._reset_pending.is_set())
        self.job._on_audio(samples, 0.5)
        self.assertEqual(self.job._queue.qsize(), 1)

    def test_stopped_capture_callback_cannot_refill_queue(self):
        self.job._stop.set()
        self.job._on_audio(np.zeros(512, dtype=np.float32), 0.0)
        self.assertTrue(self.job._queue.empty())

    def test_resume_clears_old_partial_and_resets_vad(self):
        resets = []
        self.job._partial = "古い字幕です"
        self.job._active = True
        self.job._utterance = [np.ones(512, dtype=np.float32)]
        self.job._vad = SimpleNamespace(reset=lambda: resets.append(True))
        self.job._setup = lambda: None
        self.job.pause(True)
        self.job.pause(False)
        self.job._on_audio(np.zeros(512, dtype=np.float32), 0.0)
        self.job._process_chunk = lambda *args: self.job._stop.set()
        self.job.run()
        self.assertEqual(resets, [True])
        self.assertEqual(self.job._partial, "")
        self.assertEqual(self.partials, [""])

    def test_capture_failure_ends_pipeline_and_releases_pools(self):
        statuses, watchdog_fired = [], []
        self.job.sig_status.connect(statuses.append, QtCore.Qt.DirectConnection)
        self.job.cfg.get = lambda key, default=None: {
            "asr.partial": False, "translate.enabled": False,
        }.get(key, default)
        class FailedCapture:
            def __init__(self, **kwargs):
                self.on_error = kwargs["on_error"]
            def start(self):
                self.on_error("device disconnected")
            def stop(self):
                pass
            def join(self, timeout=None):
                pass
        model = SimpleNamespace(load=lambda: None, close=lambda: None)
        sherpa = SimpleNamespace(
            VadModelConfig=lambda: SimpleNamespace(silero_vad=SimpleNamespace()),
            VoiceActivityDetector=lambda *args, **kwargs: None,
        )
        def watchdog():
            watchdog_fired.append(True)
            self.job._stop.set()
        timer = threading.Timer(.5, watchdog)
        timer.daemon = True
        # Exercise the actual startup callback. The watchdog bounds the old
        # bug, where a terminated capture leaves the main loop alive forever.
        with patch.object(pipeline, "AudioCapture", FailedCapture), patch.object(
                pipeline, "FinalASR", return_value=model), patch.object(
                pipeline, "MODELS") as model_dir, patch.dict("sys.modules", {"sherpa_onnx": sherpa}):
            model_dir.__truediv__.return_value.stat.return_value = SimpleNamespace(
                st_mode=0o100644, st_size=650_000)
            timer.start()
            try:
                self.job.run()
            finally:
                timer.cancel()
                timer.join()
        self.assertEqual(watchdog_fired, [])
        self.assertTrue(self.job._stop.is_set())
        self.assertTrue(self.job._pools_closed)
        self.assertFalse(any(text == "正在聆听…" for text in statuses))

    def test_warmup_passes_cancellation_control_and_suppresses_stale_info(self):
        infos, controls = [], []
        self.job.sig_translate_info.connect(infos.append, QtCore.Qt.DirectConnection)
        def translate(text, context, on_delta=None):
            controls.append(on_delta)
            self.job._stop.set()
            on_delta("你好")
        fake = SimpleNamespace(translate=translate)
        with patch.object(pipeline, "Translator", return_value=fake):
            self.job._warmup_translate()
        self.assertEqual(len(controls), 1)
        self.assertIsInstance(controls[0], pipeline.TranslationStream)
        self.assertEqual(infos, [])


class AudioTests(unittest.TestCase):
    def test_invalid_device_samples_are_finite_before_resampling(self):
        chunks, errors, inputs = [], [], []
        capture = audio.AudioCapture("loopback", "", lambda *args: None, errors.append)
        raw = np.array([[np.inf, 1.0], [-np.inf, -1.0], [np.nan, .4]], dtype=np.float32)
        class Recorder:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def record(self, numframes):
                return raw
        class Resampler:
            def resample_chunk(self, samples):
                inputs.append(samples.copy())
                return samples
        def on_chunk(samples, rms):
            chunks.append((samples, rms))
            capture.stop()
        capture.on_chunk = on_chunk
        device = SimpleNamespace(name="test", samplerate=48000, recorder=lambda **kw: Recorder())
        with patch.object(audio, "_pick_device", return_value=device), patch.object(
                audio.soxr, "ResampleStream", return_value=Resampler()):
            capture.run()
        self.assertEqual(errors, [])
        self.assertEqual(len(chunks), 1)
        self.assertTrue(np.isfinite(inputs[0]).all())
        np.testing.assert_allclose(inputs[0], [.5, -.5, .2])
        self.assertTrue(np.isfinite(chunks[0][1]))

    def test_clean_stereo_mix_is_preserved(self):
        chunks = []
        capture = audio.AudioCapture("loopback", "", lambda *args: None, self.fail)
        class Recorder:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def record(self, numframes):
                return np.array([[.1, .3], [-.8, .2]], dtype=np.float32)
        def on_chunk(samples, rms):
            chunks.append(samples)
            capture.stop()
        capture.on_chunk = on_chunk
        device = SimpleNamespace(name="test", samplerate=16000, recorder=lambda **kw: Recorder())
        with patch.object(audio, "_pick_device", return_value=device):
            capture.run()
        np.testing.assert_allclose(chunks[0], [.2, -.3])


if __name__ == "__main__":
    unittest.main()
