"""Long-session UI regressions; no model, device, or user data required."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtWidgets

from jpsub.core.config import Config
from jpsub.core.segment import Segment
from jpsub.ui.transcript import TranscriptView
from jpsub.ui.window import MainWindow


class NoTraversalList(list):
    def __iter__(self):
        raise AssertionError('streaming updates must not scan the session history')

    def __add__(self, other):
        raise AssertionError('streaming updates must not copy the session history')


class TranscriptLongRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = Config(Path(self.temp.name) / 'config.json')
        self.cfg.set('ui.max_lines', 10, save=False)
        self.view = TranscriptView(self.cfg)
        self.addCleanup(self.view.close)

    def test_copy_button_checks_do_not_scan_or_copy_full_history(self):
        self.view.load_segments([Segment(sid=str(i), jp='日本語') for i in range(10000)])
        self.assertEqual(len(self.view.lines), 10)
        self.view._history = NoTraversalList(self.view._history)
        for _ in range(100):
            self.assertTrue(self.view.has_content('jp'))
            self.assertTrue(self.view.has_content('both'))
            self.assertFalse(self.view.has_content('zh'))
        self.view.set_zh_for('0', '补译')
        self.assertTrue(self.view.has_content('zh'))

    def test_content_counts_follow_replacement_clear_and_hidden_history(self):
        self.cfg.set('ui.max_lines', 1, save=False)
        self.view.commit_segment(Segment(sid='old', jp='原文', zh='译文'))
        self.view.commit_segment(Segment(sid='new', jp='新句'))
        self.view.commit_segment(Segment(sid='old', jp='', zh=''))
        self.assertFalse(self.view.has_content('zh'))
        self.assertTrue(self.view.has_content('jp'))
        self.view.commit_segment(Segment(sid='new', jp='', zh=''))
        self.assertFalse(self.view.has_content())
        self.view.set_live_jp('草稿')
        self.view.set_live_zh('预览')
        self.assertTrue(self.view.has_content('zh'))
        self.view.set_live_zh('')
        self.assertFalse(self.view.has_content('zh'))
        self.view.clear_all()
        self.assertFalse(self.view.has_content())
        self.assertEqual(self.view._content_counts, [0, 0])

    def test_committing_live_draft_counts_it_once(self):
        self.view.set_live_jp('原文')
        self.view.set_live_zh('译文')
        self.view.commit_segment(Segment(sid='final', jp='原文'))
        self.assertEqual(self.view._content_counts, [1, 1])
        self.view.commit_segment(Segment(sid='final', jp='', zh=''))
        self.assertEqual(self.view._content_counts, [0, 0])

    def test_partial_translation_error_is_visible_but_not_exported(self):
        line = self.view.commit_segment(Segment(sid='one', jp='日文', zh='未完成的译文'))
        window = SimpleNamespace(transcript=self.view, _failed_translation_ids=set(),
                                 _history_dirty={}, overlay=Mock())
        MainWindow._on_translation_failed(window, 'one', '队列过载')
        self.assertIn('未完成的译文', self.view.toPlainText())
        self.assertIn('可点补译', self.view.toPlainText())
        self.assertEqual(self.view.all_segments()[0].zh, '未完成的译文')
        self.assertFalse(self.view.all_segments()[0].done)
        self.assertEqual(line.zh.tooltip, '队列过载')
        self.view.set_zh_for('one', '完整译文')
        self.assertNotIn('可点补译', self.view.toPlainText())

    def test_utterance_boundary_does_not_mix_old_chinese_with_new_japanese(self):
        self.view.set_live_jp('前の文')
        self.view.set_live_zh('上一句')
        window = SimpleNamespace(transcript=self.view, _live_jp='前の文', _live_zh='上一句',
                                 _refresh_overlay=Mock(), _ensure_session=Mock())
        MainWindow._on_partial(window, '')
        MainWindow._on_partial(window, '次の文')
        self.assertEqual(window._live_zh, '')
        self.assertNotIn('上一句', self.view.toPlainText())
        self.assertIn('次の文', self.view.toPlainText())


class HistoryFlushLongRunTests(unittest.TestCase):
    def window(self):
        return SimpleNamespace(history=Mock(), _session_id='session',
            _live_jp='草稿', _live_zh='预览', _session_start=0,
            _history_dirty={}, _history_error='', toast_msg=Mock())

    def test_unchanged_draft_does_not_fsync_every_timer_tick(self):
        window = self.window()
        for _ in range(100):
            MainWindow._flush_history(window)
        self.assertEqual(window.history.write.call_count, 1)
        window._live_zh = '更新预览'
        MainWindow._flush_history(window)
        self.assertEqual(window.history.write.call_count, 2)

    def test_empty_draft_clears_persisted_preview_once(self):
        window = self.window()
        MainWindow._flush_history(window)
        window._live_jp = window._live_zh = ''
        MainWindow._flush_history(window)
        self.assertTrue(window.history.write.call_args.kwargs['clear_draft'])
        MainWindow._flush_history(window)
        self.assertEqual(window.history.write.call_count, 2)

    def test_failed_save_is_retried_with_identical_draft(self):
        window = self.window()
        window.history.write.side_effect = [OSError('test-only disk failure'), None]
        with self.assertLogs(level='ERROR'):
            MainWindow._flush_history(window)
        MainWindow._flush_history(window)
        self.assertEqual(window.history.write.call_count, 2)

    def test_same_draft_in_new_session_is_saved(self):
        window = self.window()
        MainWindow._flush_history(window)
        window._session_id = 'new-session'
        MainWindow._flush_history(window)
        self.assertEqual(window.history.write.call_count, 2)

    def test_final_updates_are_saved_even_when_draft_unchanged(self):
        window = self.window()
        MainWindow._flush_history(window)
        window._history_dirty['one'] = Segment(sid='one', jp='日文', zh='中文', done=True)
        MainWindow._flush_history(window)
        self.assertEqual(window.history.write.call_count, 2)
        self.assertIsNone(window.history.write.call_args.kwargs['draft'])
        self.assertEqual(window._history_dirty, {})

    def test_bulk_retry_does_not_evict_live_work_and_counts_acceptances(self):
        segments = [Segment(sid=str(i), jp='日文') for i in range(20)]
        window = SimpleNamespace(pipeline=Mock(), transcript=Mock(),
                                 _failed_translation_ids=set(), toast_msg=Mock())
        window.transcript.session_segments.return_value = segments
        window.pipeline.enqueue_translation.side_effect = [True] * 8 + [False] * 12
        MainWindow._retry_missing_translations(window)
        self.assertIn('8 条补译', window.toast_msg.call_args.args[0])
        self.assertTrue(all(c.kwargs == {'retry': True}
                            for c in window.pipeline.enqueue_translation.call_args_list))


if __name__ == '__main__':
    unittest.main()
