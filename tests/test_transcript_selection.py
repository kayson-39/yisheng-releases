"""Synthetic subtitle delivery only: no device, recognizer, visible window or user files."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PySide6 import QtGui, QtWidgets
from jpsub.core.pipeline import Pipeline
from jpsub.core.segment import Segment
from jpsub.ui.transcript import TranscriptView
from jpsub.ui.window import MainWindow


class SelectionStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.view = TranscriptView({'ui.max_lines': 10})
        self.addCleanup(self.view.close)
        self.window = SimpleNamespace(transcript=self.view, _live_jp='', _live_zh='',
            _ensure_session=Mock(), _refresh_overlay=Mock(), lbl_count=Mock(),
            _history_dirty={}, _flush_history=Mock(), sender=lambda: None, pipeline=object(),
            _session_timeline=SimpleNamespace(bind=lambda _: 0.0, last_end=0.0))

    def hold_first(self):
        for i in range(10):
            self.view.commit_segment(Segment(str(i), 'Japanese ' + str(i)))
        cursor = self.view.textCursor()
        cursor.setPosition(0)
        cursor.setPosition(3, QtGui.QTextCursor.MoveMode.KeepAnchor)
        self.view.setTextCursor(cursor)

    def test_stream_update_checks_only_its_own_block(self):
        self.hold_first()
        with patch.object(self.view, '_selected_lines', side_effect=AssertionError('full scan')):
            self.view.set_zh_for('9', 'current translation')
            self.view.set_zh_for('0', 'selected translation')
        self.assertEqual(self.view.textCursor().selectedText(), 'Jap')
        self.assertIn('current translation', self.view.toPlainText())
        self.assertNotIn('selected translation', self.view.toPlainText())
        self.view.jump_latest()
        self.assertIn('selected translation', self.view.toPlainText())

    def test_held_reading_window_is_bounded_and_catches_up_from_complete_history(self):
        self.hold_first()
        for i in range(10, 110):
            self.view.set_live_jp('preview ' + str(i))
            self.view.commit_segment(Segment(str(i), 'Japanese ' + str(i), live_jp='preview ' + str(i)))
        self.view.set_zh_for('108', 'older late translation')
        self.view.set_live_jp('next draft')
        self.assertLessEqual(len(self.view.lines), 11)
        self.assertLessEqual(self.view.document().blockCount(), 12)
        self.assertEqual(self.view.textCursor().selectedText(), 'Jap')
        self.assertEqual(len(self.view.all_segments()), 110)
        self.view.jump_latest()
        self.assertEqual([r.seg.sid for r in self.view.lines], [str(i) for i in range(100, 110)])
        self.assertIn('older late translation', self.view.toPlainText())
        self.assertIn('next draft', self.view.toPlainText())

    def test_boundary_actually_removes_bilingual_draft_and_leaves_no_blank_rows(self):
        self.view.commit_segment(Segment('saved', 'saved source', 'saved translation'))
        for _ in range(30):
            MainWindow._on_partial(self.window, 'false draft')
            self.view.set_live_zh('false translation')
            MainWindow._on_partial(self.window, '')
        self.assertEqual(self.view.document().blockCount(), 1)
        self.assertEqual(self.view.toPlainText(), 'saved source\nsaved translation')
        self.assertIsNone(self.view._live)
        self.assertEqual(len(self.view.all_segments()), 1)

    def test_catch_up_displays_draft_even_before_another_final_arrives(self):
        self.hold_first()
        self.view.set_live_jp('still talking')
        self.assertNotIn('still talking', self.view.toPlainText())
        self.view.jump_latest()
        self.assertIn('still talking', self.view.toPlainText())
        self.assertEqual(len(self.view.all_segments()), 10)

    def test_empty_session_does_not_leave_an_extra_block_or_copyable_draft(self):
        MainWindow._on_partial(self.window, 'temporary')
        MainWindow._on_partial(self.window, '')
        self.assertEqual(self.view.toPlainText(), '')
        self.assertFalse(self.view.has_content('both'))
        self.view.commit_segment(Segment('first', 'first real sentence'))
        self.assertEqual(self.view.toPlainText(), 'first real sentence')

    def test_stale_final_cannot_claim_repeated_word_in_new_draft(self):
        MainWindow._on_partial(self.window, 'はい')
        MainWindow._on_partial(self.window, '')
        MainWindow._on_partial(self.window, 'はい')
        MainWindow._on_segment(self.window, Segment('first', 'はい', live_jp='はい'))
        self.assertIsNotNone(self.view._live)
        self.assertEqual(self.view._live.seg.jp, 'はい')
        self.assertEqual(len(self.view.session_segments()), 1)

    def test_japanese_revision_clears_obsolete_chinese_but_extension_does_not(self):
        MainWindow._on_partial(self.window, '彼は')
        self.window._live_zh = '他'
        self.view.set_live_zh('他')
        with patch.object(self.view, 'set_live_zh', wraps=self.view.set_live_zh) as update_zh:
            MainWindow._on_partial(self.window, '彼は今日')
            self.assertEqual(self.view._live.seg.zh, '他')
            MainWindow._on_partial(self.window, '彼女は今日')
            self.assertEqual(self.window._live_zh, '')
            self.assertEqual(self.view._live.seg.zh, '')
            self.assertNotIn('他', self.view.toPlainText())
            MainWindow._on_partial(self.window, '私は')
            MainWindow._on_partial(self.window, '')
            update_zh.assert_called_once_with('')

    def test_late_empty_final_does_not_clear_newer_preview(self):
        pipe = SimpleNamespace(_stop=Mock(), sig_partial=Mock())
        pipe._stop.is_set.return_value = False
        Pipeline._emit_final(pipe, Segment('old', ''), SimpleNamespace(text=''))
        pipe.sig_partial.emit.assert_not_called()
