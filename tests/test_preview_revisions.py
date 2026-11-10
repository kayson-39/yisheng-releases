"""One-slot live preview scheduler. No devices, GUI, model, network or sleeps."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import Qt
from jpsub.core import pipeline


class Config(dict):
    def service(self, provider):
        return {'model': 'galtransl'}


class PreviewRevisions(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        clock = patch.object(pipeline.time, 'monotonic', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.q = pipeline.Pipeline(Config({'translate.provider': 'llamacpp',
            'translate.live_provider': 'same', 'translate.live_min_interval_ms': 500}))
        self.q._active = True
        self.q._live.update(sid='L1', gen=1)
        self.calls, self.updates = [], []
        self.q._submit = lambda pool, fn, *args: self.calls.append((fn,args)) or True
        self.q.sig_live_zh.connect(lambda s,t: self.updates.append((s,t)), Qt.DirectConnection)
        self.empty = np.zeros(0, dtype=np.float32)
        self.first = '今日は映画を見ます'
        self.latest = '明日は家で料理を作ります'

    def tearDown(self):
        self.q._stop.set()
        self.q._shutdown_pools()

    def tick(self):
        self.q._process_chunk(self.empty, 0.)

    def draft(self, text):
        self.q._draft_engine = SimpleNamespace(transcribe=lambda a: SimpleNamespace(text=text))
        self.q._run_draft(self.q._live['gen'], self.empty)
        self.tick()

    def execute(self, index, fn):
        worker,args = self.calls[index]
        translator = SimpleNamespace(translate=fn, last_provider='llamacpp')
        with patch.object(pipeline,'Translator',return_value=translator):
            worker(*args)

    def test_burst_while_busy_is_coalesced_and_stable_latest_is_translated(self):
        self.draft(self.first)
        def old_request(t, ctx, provider, on_delta):
            for n in range(100):
                self.draft(self.latest + str(n))
            self.assertEqual(len(self.calls), 1)
            on_delta('旧结果')
            return '旧结果'
        self.execute(0,old_request)
        self.assertFalse(any(t == '旧结果' for s,t in self.updates))
        self.tick()
        self.assertEqual(len(self.calls), 1)
        self.now += .51
        self.tick()  # No further ASR text changes.
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[-1][1][-1], self.latest+'99')
        self.execute(1,lambda *a,**kw:'新译文')
        self.assertEqual(self.updates[-1][1], '新译文')
        self.now += 20
        self.tick()
        self.assertEqual(len(self.calls), 2)

    def test_equal_length_shorter_and_one_character_corrections(self):
        self.draft(self.first)
        for index, revised in enumerate(('明日は料理をします','家で料理します','家で料理しますね')):
            self.execute(index,lambda *a,**kw:'译文')
            old_sid = self.updates[-1][0]
            self.draft(revised)
            # Queued Qt updates from the old revision can no longer pass the UI check.
            self.assertNotEqual(self.q._live['sid'], old_sid)
            self.now += .51
            self.tick()
            self.assertEqual(self.calls[index+1][1][-1], revised)

    def test_short_first_draft_is_filtered_but_short_correction_replaces_old_meaning(self):
        self.draft('はい')
        self.assertEqual(self.calls, [])
        self.draft(self.first)
        self.execute(0,lambda *a,**kw:'今天看电影')
        self.draft('違います')
        self.now += .51
        self.tick()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.calls[1][1][-1], '違います')

    def test_final_preemption_does_not_starve_a_stable_draft(self):
        self.draft(self.first)
        self.q._pending_final['final'] = self.now
        entered = []
        self.execute(0,lambda *a,**kw:entered.append(True) or '不应送达')
        self.assertFalse(entered)
        self.now += 1
        self.tick()
        self.assertEqual(len(self.calls), 1)
        self.q._pending_final.clear()
        self.tick()
        self.assertEqual(len(self.calls), 2)
        self.execute(1,lambda *a,**kw:'最新')
        self.assertEqual(self.updates[-1][1],'最新')

    def lifecycle_case(self, change):
        self.draft(self.first)
        def old_request(*a,**kw):
            self.draft(self.latest)
            change()
            return '停止后不许发出'
        self.execute(0,old_request)
        self.now += 20
        self.tick()
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(any(t == '停止后不许发出' for s,t in self.updates))

    def test_queued_revision_dies_on_stop(self):
        self.lifecycle_case(self.q._stop.set)

    def test_queued_revision_dies_on_boundary(self):
        self.lifecycle_case(self.q._reset_utterance)

    def test_queued_revision_dies_on_quick_pause_resume(self):
        self.lifecycle_case(lambda: (self.q.pause(True), self.q.pause(False)))
