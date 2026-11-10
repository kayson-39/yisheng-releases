"""Real SQLite locks in a private in-memory database; no personal history or UI."""
import sqlite3
import threading
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QMessageBox
from jpsub.core.history import HistoryStore
from jpsub.core.segment import Segment
from jpsub.ui.window import MainWindow


class HistoryContention(unittest.TestCase):
    def setUp(self):
        self.store = HistoryStore.__new__(HistoryStore)
        uri = f'file:/test-{uuid.uuid4().hex}?vfs=memdb'
        self.store._lock = threading.RLock()
        self.store.db = sqlite3.connect(uri, uri=True, timeout=0.8)
        self.addCleanup(self.store.close)
        self.store.db.executescript("""
            CREATE TABLE sessions(id TEXT PRIMARY KEY,title TEXT,started REAL,updated REAL,state TEXT);
            CREATE TABLE items(session TEXT,sid TEXT,position INTEGER,payload TEXT,PRIMARY KEY(session,sid));
        """)
        self.sid = self.store.begin()
        self.other = sqlite3.connect(uri, uri=True)
        self.addCleanup(self.other.close)

    def test_busy_autosave_returns_promptly_and_retries_the_original_caption(self):
        self.other.execute('BEGIN IMMEDIATE')
        w = SimpleNamespace(history=self.store, _session_id=self.sid, _live_jp='', _live_zh='',
            _history_dirty={'one': Segment('one', 'test')}, _history_error='', toast_msg=Mock())
        t0 = time.monotonic()
        with patch('logging.exception'):
            self.assertFalse(MainWindow._flush_history(w))
        self.assertLess(time.monotonic()-t0, 0.4)
        self.assertEqual(list(w._history_dirty), ['one'])
        self.other.rollback()
        self.assertTrue(MainWindow._flush_history(w, state='stopped'))
        self.assertEqual(w._history_dirty, {})
        self.assertEqual(self.store.db.execute('SELECT session,sid FROM items').fetchall(),
                         [(self.sid,'one')])
        self.assertEqual(self.store.db.execute('PRAGMA busy_timeout').fetchone()[0], 800)

    def test_busy_begin_does_not_freeze_ui_or_create_an_empty_duplicate(self):
        self.other.execute('BEGIN IMMEDIATE')
        t0 = time.monotonic()
        with self.assertRaises(sqlite3.OperationalError):
            self.store.begin()
        self.assertLess(time.monotonic()-t0, 0.4)
        self.other.rollback()
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 1)
        self.assertNotEqual(self.store.begin(), self.sid)


class CloseOnSaveFailure(unittest.TestCase):
    def window(self):
        w=SimpleNamespace(pipeline=None, _retiring_pipelines=[], _export_jobs=[], text_translation=Mock(),
            _flush_history=Mock(return_value=False), _history_timer=Mock(), _history_dialog=None,
            _export_dialog=None, overlay=None, tray=None, _save_geometry=Mock(), history=Mock(),
            _on_status=Mock())
        w.text_translation.is_busy.return_value=False
        return w

    def test_cancel_keeps_window_and_unsaved_session_available(self):
        w=self.window(); ev=Mock(); db=w.history
        with patch.object(QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Cancel) as dialog, patch('jpsub.core.llama_engine.stop_all'):
            MainWindow.closeEvent(w, ev)
        dialog.assert_called_once()
        self.assertEqual(dialog.call_args.args[-1], QMessageBox.StandardButton.Cancel)
        ev.ignore.assert_called_once(); ev.accept.assert_not_called()
        db.close.assert_not_called(); w._history_timer.stop.assert_not_called()
        self.assertFalse(w._closing)
        w.text_translation.resume_after_cancelled_shutdown.assert_called_once()

    def test_only_explicit_discard_allows_exit_after_save_failure(self):
        w=self.window(); ev=Mock(); db=w.history
        with patch.object(QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Discard) as dialog, patch('jpsub.core.llama_engine.stop_all'):
            MainWindow.closeEvent(w, ev)
        dialog.assert_called_once(); ev.accept.assert_called_once(); db.close.assert_called_once()

    def test_successful_save_needs_no_confirmation(self):
        w=self.window(); w._flush_history.return_value=True; ev=Mock()
        with patch.object(QMessageBox, 'warning') as dialog, patch('jpsub.core.llama_engine.stop_all'):
            MainWindow.closeEvent(w, ev)
        dialog.assert_not_called(); ev.accept.assert_called_once()
