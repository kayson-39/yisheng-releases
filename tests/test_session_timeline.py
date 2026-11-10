"""Check the real session/restart/export paths. No GUI, recording or model jobs."""
import unittest
from unittest.mock import Mock, patch
import numpy as np

from jpsub.core.export import to_srt
from jpsub.core.pipeline import Pipeline
from jpsub.core.segment import Segment
from jpsub.ui.window import MainWindow


class SessionTimelineTests(unittest.TestCase):
    def setUp(self):
        self.jobs = []
        w = self.w = Mock()
        w.cfg = {}; w.pipeline = None; w._closing = False; w._retiring_pipelines = []
        w._preserve_session_for_restart = False; w._session_id = None
        w._history_dirty = {}; w.overlay = None
        w.text_translation.is_busy.return_value = False
        w.sender.return_value = None
        self.saved = []
        w.transcript._live = None; w.transcript._history_by_sid = {}
        def save(s, **kwargs):
            self.saved.append(s)
            w.transcript._history_by_sid[s.sid] = s
        w.transcript.commit_segment.side_effect = save
        w.transcript.session_segments.side_effect = lambda: self.saved
        w.transcript.clear_all.side_effect = self.saved.clear
        w._stop_pipeline.side_effect = lambda: setattr(w, 'pipeline', None)
        w._ensure_session.side_effect = lambda: setattr(w, '_session_id', w._session_id or 'S')
        with patch.object(Pipeline, 'start'):
            MainWindow._start_pipeline(w)
        self.jobs.append(w.pipeline)
        self.addCleanup(self.release)

    def release(self):
        for p in self.jobs:
            p._shutdown_pools()
            p._release_resources()

    def audio(self, begins_at, duration=0.08):
        with patch('jpsub.core.pipeline.time.monotonic', return_value=begins_at + duration):
            self.w.pipeline._on_audio(np.zeros(round(16000*duration), dtype=np.float32), 0.0)

    def segment(self, start, end):
        source = Segment(str(len(self.saved)), 'subtitle', t0=start, t1=end)
        MainWindow._on_segment(self.w, source)
        self.assertEqual((source.t0, source.t1), (start, end))  # ASR owns its local object.
        return self.saved[-1]

    def restart(self, preserve=True):
        if preserve:
            with patch.object(Pipeline, 'isRunning', return_value=True):
                MainWindow._restart_if_running(self.w)
        else:
            self.w.pipeline = None
        with patch.object(Pipeline, 'start'):
            MainWindow._start_pipeline(self.w)
        self.jobs.append(self.w.pipeline)

    def test_device_restart_preserves_srt_clock_including_time_without_capture(self):
        self.audio(100)
        self.segment(600, 600.64)
        self.restart()
        self.audio(710)
        self.segment(2, 2.64)
        self.assertEqual(self.w._session_id, 'S')
        srt = to_srt(self.saved)
        self.assertIn('00:10:00,000 --> 00:10:00,640', srt)
        self.assertIn('00:10:12,000 --> 00:10:12,640', srt)

    def test_repeated_restarts_do_not_double_add_elapsed_time(self):
        for i, epoch in enumerate((100,120,130)):
            if i: self.restart()
            self.audio(epoch)
            self.segment(1,2)
        self.assertEqual([s.t0 for s in self.saved], [1,21,31])

    def test_silent_first_epoch_sets_origin_before_any_recognized_sentence(self):
        self.audio(10)
        self.restart()
        self.audio(40)
        self.assertEqual(self.segment(1,2).t0,31)

    def test_capture_not_model_warmup_is_origin_and_new_session_starts_fresh(self):
        self.restart()  # No audio ever arrived before this restart.
        self.audio(500)
        self.assertEqual(self.segment(1,2).t0,1)
        self.restart(preserve=False)
        self.audio(900)
        self.assertEqual(self.segment(3,4).t0,3)
        self.assertEqual(len(self.saved), 1)

    def test_clear_while_listening_establishes_new_session_zero(self):
        self.audio(100,2)
        MainWindow._clear(self.w)
        MainWindow._on_segment(self.w, Segment('old', 'pre-clear audio', t0=.5,t1=1))
        self.assertFalse(self.saved)
        crossing=self.segment(1.9,2.1)
        self.assertAlmostEqual(crossing.t0,0)
        self.assertAlmostEqual(crossing.t1,.1)
        self.assertEqual(self.segment(2.5,3.5).t0,.5)
        self.restart()
        self.audio(120)
        self.assertEqual(self.segment(1,2).t0,19)

    def test_new_epoch_starts_no_earlier_than_preceding_audio_endpoint(self):
        self.audio(100)
        self.segment(19,20)
        self.restart()
        self.audio(119.5)  # A burst made the old capture position lead its wall clock.
        self.assertEqual(self.segment(0,.5).t0,20)
        self.assertEqual(self.segment(1,1.5).t0,21)

    def test_saved_draft_uses_same_clock_as_committed_subtitles(self):
        self.audio(100)
        self.segment(5,6)
        self.restart()
        self.audio(110,2)
        w=self.w
        w._live_jp='draft'; w._live_zh=''; w._history_saved_draft=None
        MainWindow._flush_history(w)
        draft=w.history.write.call_args.kwargs['draft']
        self.assertEqual((draft.t0,draft.t1),(12,12))

    def test_queued_signal_from_retired_pipeline_cannot_change_new_timeline(self):
        self.audio(100)
        old=self.w.pipeline
        self.restart()
        self.audio(130)
        self.w.sender.return_value=old
        MainWindow._on_segment(self.w, Segment('late','stale',t0=50,t1=51))
        self.assertFalse(self.saved)
        self.w.sender.return_value=self.w.pipeline
        self.assertEqual(self.segment(1,2).t0,31)
