"""History polling must not repeatedly parse unchanged archives on the UI thread."""
import sqlite3
import unittest
import uuid
from unittest.mock import Mock

from PySide6 import QtCore, QtWidgets

from jpsub.core.segment import Segment
from jpsub.core.history import HistoryStore
from jpsub.ui.history import HistoryDialog


class FakeTranscript(QtCore.QObject):
    content_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.segments = [Segment('live', 'live original')]
        self.all_segments = Mock(side_effect=lambda: list(self.segments))


class HistoryPollingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self):
        self.rows = [dict(id='saved', title='Saved', count=1, state='stopped', updated=1)]
        self.saved = [Segment('one', 'original')]
        self.owner = QtWidgets.QWidget()
        self.owner.transcript = FakeTranscript()
        self.owner.history = Mock()
        self.owner.history.sessions.side_effect = lambda: [dict(r) for r in self.rows]
        self.owner.history.change_token.side_effect = lambda: (tuple(tuple(r.items()) for r in self.rows), 0)
        self.owner.history.load.side_effect = lambda sid: list(self.saved)
        self.owner._export_segments = Mock()
        self.dialog = HistoryDialog(self.owner)
        self.dialog.timer.stop()
        self.addCleanup(self.owner.close)
        self.addCleanup(self.dialog.close)

    def select_saved(self):
        self.dialog.list.setCurrentRow(1)
        self.owner.history.load.reset_mock()

    def test_unchanged_saved_session_is_not_loaded_each_timer_tick(self):
        self.select_saved()
        for _ in range(20):
            self.dialog.refresh()
        self.owner.history.load.assert_not_called()
        self.assertIn('original', self.dialog.text.toPlainText())

    def test_idle_poll_does_not_recount_all_saved_subtitles(self):
        self.owner.history.sessions.reset_mock()
        for _ in range(20):
            self.dialog.refresh()
        self.owner.history.sessions.assert_not_called()
        self.rows[0]['count'] = 2
        self.dialog.refresh()
        self.owner.history.sessions.assert_called_once_with()
        self.assertIn('2 条字幕', self.dialog.list.item(1).text())

    def test_unchanged_live_session_is_not_copied_each_timer_tick(self):
        self.owner.transcript.all_segments.reset_mock()
        for _ in range(20):
            self.dialog.refresh()
        self.owner.transcript.all_segments.assert_not_called()

    def test_translation_update_with_same_count_invalidates_saved_cache(self):
        self.select_saved()
        self.saved[0].zh = 'new translation'
        self.rows[0]['updated'] = 2
        self.dialog.refresh()
        self.owner.history.load.assert_called_once_with('saved')
        self.assertIn('new translation', self.dialog.text.toPlainText())

    def test_live_change_signal_invalidates_without_count_change(self):
        self.owner.transcript.all_segments.reset_mock()
        self.owner.transcript.segments[0].zh = 'new live translation'
        self.owner.transcript.content_changed.emit()
        self.dialog.refresh()
        self.owner.transcript.all_segments.assert_called_once_with()
        self.assertIn('new live translation', self.dialog.text.toPlainText())

    def test_external_commit_without_metadata_change_invalidates_saved_cache(self):
        self.select_saved()
        self.saved[0].zh = 'another connection updated the payload'
        self.owner.history.change_token.side_effect = None
        self.owner.history.change_token.return_value = (0, 1)
        self.dialog.refresh()
        self.owner.history.load.assert_called_once_with('saved')
        self.assertIn(self.saved[0].zh, self.dialog.text.toPlainText())

    def test_unchanged_global_search_does_not_reload_all_archives(self):
        self.dialog.scope.setCurrentIndex(1)
        self.dialog.query.setText('original')
        self.owner.history.load.reset_mock()
        for _ in range(20):
            self.dialog.refresh()
        self.owner.history.load.assert_not_called()
        self.assertIn('【Saved】', self.dialog.text.toPlainText())

    def test_query_change_and_rename_refresh_global_search(self):
        self.dialog.scope.setCurrentIndex(1)
        self.dialog.query.setText('does-not-match')
        self.assertNotIn('original', self.dialog.text.toPlainText())
        self.dialog.query.setText('original')
        self.assertIn('original', self.dialog.text.toPlainText())
        self.rows[0]['title'] = 'Renamed'
        self.dialog.refresh()
        self.assertIn('【Renamed】', self.dialog.text.toPlainText())

    def test_changing_store_with_same_metadata_does_not_keep_old_text(self):
        self.select_saved()
        replacement = Mock()
        replacement.sessions.side_effect = lambda: [dict(r) for r in self.rows]
        replacement.load.return_value = [Segment('one', 'another store')]
        self.owner.history = replacement
        self.dialog.refresh()
        self.assertIn('another store', self.dialog.text.toPlainText())


class HistoryChangeTokenTests(unittest.TestCase):
    def test_token_tracks_same_and_other_connections_without_rescanning_items(self):
        uri = f'file:{uuid.uuid4().hex}?mode=memory&cache=shared'
        store = HistoryStore.__new__(HistoryStore)
        store.db = sqlite3.connect(uri, uri=True)
        self.addCleanup(store.db.close)
        external = sqlite3.connect(uri, uri=True)
        self.addCleanup(external.close)
        store.db.execute('CREATE TABLE t (caption TEXT)')
        before = store.change_token()
        with store.db:
            store.db.execute("INSERT INTO t VALUES ('first')")
        local = store.change_token()
        self.assertNotEqual(before, local)
        with external:
            external.execute("UPDATE t SET caption='translated'")
        after = store.change_token()
        self.assertNotEqual(local, after)
        store.db.execute('SELECT * FROM t').fetchall()
        self.assertEqual(store.change_token(), after)


if __name__ == '__main__':
    unittest.main()
