"""Session boundaries must not move unsaved captions into another session."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jpsub.core.segment import Segment
from jpsub.ui.window import MainWindow


class SessionSaveReliabilityTests(unittest.TestCase):
    def window(self):
        window = SimpleNamespace(
            _closing=False, text_translation=Mock(), _retiring_pipelines=[],
            pipeline=None, _preserve_session_for_restart=False,
            _session_id='original-session', _session_start=0,
            _history_dirty={'old': Segment(sid='old', jp='必须留在原场次')},
            _history_error='', _live_jp='', _live_zh='', history=Mock(),
            transcript=Mock(), _failed_translation_ids=set(), overlay=None,
            lbl_count=Mock(), toast_msg=Mock(), _refresh_overlay=Mock(),
        )
        window.text_translation.is_busy.return_value = False
        window._flush_history = lambda **kw: MainWindow._flush_history(window, **kw)
        return window

    def test_flush_reports_failure_without_discarding_dirty_captions(self):
        window = self.window()
        window.history.write.side_effect = OSError('test-only unavailable disk')
        with self.assertLogs(level='ERROR'):
            self.assertIs(MainWindow._flush_history(window), False)
        self.assertIn('old', window._history_dirty)

    def test_flush_reports_success_and_clears_dirty_after_commit(self):
        window = self.window()
        self.assertIs(MainWindow._flush_history(window), True)
        self.assertFalse(window._history_dirty)
        self.assertEqual(window.history.write.call_args.args[0], 'original-session')
        self.assertIs(MainWindow._flush_history(window), True)

    def test_start_refuses_to_assign_unsaved_captions_to_new_session(self):
        window = self.window()
        window.history.write.side_effect = OSError('test-only unavailable disk')
        with patch('jpsub.core.pipeline.Pipeline') as pipeline:
            with self.assertLogs(level='ERROR'):
                MainWindow._start_pipeline(window)
            pipeline.assert_not_called()
        window.transcript.clear_all.assert_not_called()
        self.assertEqual(window._session_id, 'original-session')
        self.assertIn('old', window._history_dirty)

    def test_clear_keeps_original_session_when_final_save_fails(self):
        window = self.window()
        window.history.write.side_effect = OSError('test-only unavailable disk')
        with self.assertLogs(level='ERROR'):
            MainWindow._clear(window)
        window.transcript.clear_all.assert_not_called()
        self.assertEqual(window._session_id, 'original-session')
        self.assertIn('old', window._history_dirty)

    def test_clear_succeeds_after_storage_recovers(self):
        window = self.window()
        window.history.write.side_effect = [OSError('test-only unavailable disk'), None]
        with self.assertLogs(level='ERROR'):
            MainWindow._clear(window)
        self.assertEqual(window._session_id, 'original-session')
        MainWindow._clear(window)
        self.assertIsNone(window._session_id)
        self.assertFalse(window._history_dirty)
        window.transcript.clear_all.assert_called_once()
        self.assertTrue(all(c.args[0] == 'original-session'
                            for c in window.history.write.call_args_list))


if __name__ == '__main__':
    unittest.main()
