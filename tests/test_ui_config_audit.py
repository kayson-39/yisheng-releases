"""Persistence, import/export and Qt rendering regressions; isolated files only."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6 import QtCore, QtWidgets
from PIL import Image
from jpsub.core.config import Config
from jpsub.core.glossary import read_glossary, write_glossary
from jpsub.core.history import HistoryStore
from jpsub.core.export import snapshot, write_export
from jpsub.core.segment import Segment
from jpsub.ui import background
from jpsub.ui.transcript import TranscriptView
from jpsub.ui.window import MainWindow


class IsolatedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.path = self.directory / 'config.json'

    def config(self, data):
        self.path.write_text(json.dumps(data), encoding='utf-8')
        return Config(self.path)


class ConfigTests(IsolatedTests):
    def test_bad_leaf_preserves_valid_settings(self):
        cfg = self.config({'model_setup_version': 5, 'ui': {'jp_size': None, 'theme': 'dark'},
                           'translate': {'provider': 'custom', 'services': {
                               'custom': {'api_key': 'test-only', 'model': 'my-model'}}}})
        self.assertEqual(cfg.get('ui.jp_size'), 13)
        self.assertEqual(cfg.get('ui.theme'), 'dark')
        self.assertEqual(cfg.service('custom')['api_key'], 'test-only')
        self.assertEqual(cfg.service('custom')['model'], 'my-model')

    def test_bad_service_container_recovers_without_crashing(self):
        cfg = self.config({'model_setup_version': 5, 'translate': {'services': {'custom': 'broken'}}})
        self.assertEqual(cfg.service('custom')['base_url'], '')
        cfg.set_service('custom', model='my-model')
        self.assertEqual(Config(self.path).service('custom')['model'], 'my-model')

    def test_bad_section_keeps_other_sections(self):
        cfg = self.config({'model_setup_version': 5, 'ui': [], 'audio': {'device': 'my-device'}})
        self.assertEqual(cfg.get('ui.jp_size'), 13)
        self.assertEqual(cfg.get('audio.device'), 'my-device')

    def test_recovered_backup_runs_version_migrations(self):
        self.path.write_text('{broken', encoding='utf-8')
        self.path.with_suffix('.json.bak').write_text(json.dumps({
            'model_setup_version': 3, 'ui': {'theme': 'dark'},
            'translate': {'provider': 'llamacpp', 'services': {'llamacpp': {'model': 'models/llm/Hy-MT2-1.8B-Q4_K_M.gguf'}}},
        }), encoding='utf-8')
        cfg = Config(self.path)
        self.assertEqual(cfg.get('model_setup_version'), 5)
        self.assertEqual(cfg.get('ui.theme'), 'dark')
        self.assertIn('Sakura', cfg.service('llamacpp')['model'])

    def test_missing_primary_recovers_backup(self):
        self.path.with_suffix('.json.bak').write_text(json.dumps({
            'model_setup_version': 5, 'ui': {'theme': 'dark'},
        }), encoding='utf-8')
        self.assertEqual(Config(self.path).get('ui.theme'), 'dark')

    def test_glossary_valid_rows_survive_malformed_rows(self):
        cfg = self.config({'model_setup_version': 5, 'translate': {'glossary': [
            ['配信', '直播'], 'broken', ['invalid'], [None, 'wrong'], ['配信', '直播'],
        ]}})
        self.assertEqual(cfg.get('translate.glossary'), [['配信', '直播']])

    def test_custom_model_and_unknown_settings_round_trip(self):
        data = {'model_setup_version': 3, 'future_option': {'value': 'keep'},
                'translate': {'provider': 'custom', 'services': {'custom': {
                    'base_url': 'http://localhost:12345/v1', 'model': 'my-model', 'api_key': 'test-only'}}}}
        cfg = self.config(data)
        cfg.save()
        reopened = Config(self.path)
        self.assertEqual(reopened.service('custom')['model'], 'my-model')
        self.assertEqual(reopened.get('future_option'), {'value': 'keep'})

    def test_failed_atomic_save_preserves_existing_config(self):
        cfg = self.config({'model_setup_version': 5, 'ui': {'theme': 'dark'}})
        before = self.path.read_bytes()
        cfg.set('ui.theme', 'light', save=False)
        with patch('jpsub.core.config.os.replace', side_effect=OSError('locked')):
            with self.assertRaises(OSError):
                cfg.save()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.directory.glob('.config-*.tmp')), [])


class GlossaryTests(IsolatedTests):
    def test_tsv_round_trip_with_commas_and_tabs(self):
        rows = [['はい、どうぞ', '请，请用'], ['特殊\t用語', '特别术语']]
        path = self.directory / 'terms.tsv'
        write_glossary(path, rows)
        self.assertEqual(read_glossary(path), rows)

    def test_json_top_level_glossary_does_not_evaluate_invalid_fallback(self):
        path = self.directory / 'terms.json'
        path.write_text(json.dumps({'glossary': [['配信', '直播']], 'translate': None}), encoding='utf-8')
        self.assertEqual(read_glossary(path), [['配信', '直播']])

    def test_json_and_csv_round_trip(self):
        rows = [['配信', '直播'], ['鳩行為', '跨直播间传话']]
        for suffix in ('.json', '.csv'):
            path = self.directory / ('terms' + suffix)
            write_glossary(path, rows)
            self.assertEqual(read_glossary(path), rows)


class QtTests(IsolatedTests):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_replacing_same_filename_invalidates_background(self):
        path = self.directory / 'custom-photo.png'
        layer = background.BackgroundLayer()
        with patch.object(background, 'CACHE', self.directory / 'cache'):
            Image.new('RGB', (16, 16), 'red').save(path)
            old_mtime = path.stat().st_mtime_ns
            red = layer._blurred_pixmap(path, QtCore.QSize(32, 32), 0)
            Image.new('RGB', (16, 16), 'blue').save(path)
            os.utime(path, ns=(old_mtime + 2000000000, old_mtime + 2000000000))
            blue = layer._blurred_pixmap(path, QtCore.QSize(32, 32), 0)
        self.assertGreater(red.toImage().pixelColor(8, 8).red(), 200)
        self.assertGreater(blue.toImage().pixelColor(8, 8).blue(), 200)
        layer.close()

    def test_repeated_device_refresh_only_triggers_one_switch(self):
        combo = QtWidgets.QComboBox()
        callback = Mock()
        window = SimpleNamespace(cmb_device=combo, cfg=Config(self.path), _on_device_quick=callback)
        with patch('jpsub.core.audio.list_devices', return_value={'loopback': ['A', 'B'], 'mic': []}):
            for _ in range(5):
                MainWindow._refresh_devices(window)
        combo.setCurrentIndex(1)
        callback.assert_called_once_with(1)
        combo.close()

    def test_visible_line_limit_keeps_history_and_unicode(self):
        cfg = Config(self.path)
        cfg.set('ui.max_lines', 1, save=False)
        view = TranscriptView(cfg)
        view.commit_segment(Segment(sid='one', jp='日本語😀', zh='中文😀'))
        view.commit_segment(Segment(sid='two', jp='次の文', zh='下一句'))
        self.assertEqual(len(view.lines), 1)
        self.assertEqual(len(view.session_segments()), 2)
        view.set_zh_for('one', '更新😀')
        self.assertEqual(view.all_segments()[0].zh, '更新😀')
        cfg.set('ui.max_lines', 0, save=False)
        view.set_max_lines()
        self.assertEqual(len(view.lines), 2)
        self.assertIn('更新😀', view.toPlainText())
        view.close()


class EngineResetTests(unittest.TestCase):
    def fake_window(self):
        return SimpleNamespace(
            pipeline=None, _retiring_pipelines=[], _pending_translation_engine_reset=True,
            text_translation=SimpleNamespace(is_busy=lambda: False, set_audio_active=Mock(), refresh_model_label=Mock()),
            btn_start=Mock(), _restart_requested=False, _closing=False,
            _restart_if_running=Mock(), _sync_text_translation_state=Mock(),
            _update_trans_label_from_cfg=Mock(),
        )

    def test_engine_reset_waits_for_retiring_audio_workers(self):
        window = self.fake_window()
        window._retiring_pipelines.append(object())
        with patch('jpsub.core.llama_engine.stop_translation') as stop:
            MainWindow._sync_text_translation_state(window)
        stop.assert_not_called()
        self.assertTrue(window._pending_translation_engine_reset)

    def test_engine_reset_runs_after_workers_finish(self):
        window = self.fake_window()
        with patch('jpsub.core.llama_engine.stop_translation') as stop:
            MainWindow._sync_text_translation_state(window)
        stop.assert_called_once()
        self.assertFalse(window._pending_translation_engine_reset)

    def test_engine_setting_change_retires_audio_even_without_text_job(self):
        window = self.fake_window()
        with patch('jpsub.core.llama_engine.stop_translation') as stop:
            MainWindow._on_setting_changed(window, 'translate.services.llamacpp.variant', 'cpu')
        window._restart_if_running.assert_called_once()
        window._sync_text_translation_state.assert_called_once()
        stop.assert_not_called()


class HistoryExportTests(IsolatedTests):
    def test_history_migration_does_not_resurrect_draft(self):
        first = HistoryStore(self.directory / 'one')
        self.addCleanup(first.close)
        sid = first.begin()
        first.write(sid, draft=Segment(sid='__draft__', jp='draft'))
        second = first.migrate_to(self.directory / 'two')
        self.addCleanup(second.close)
        second.write(sid, [Segment(sid='final', jp='日本語', zh='中文', done=True)], clear_draft=True)
        returned = second.migrate_to(self.directory / 'one')
        self.addCleanup(returned.close)
        self.assertEqual([item.sid for item in returned.load(sid)], ['final'])

    def test_export_snapshot_is_immutable_and_atomic(self):
        segment = Segment(sid='one', jp='日本語', zh='中文', t0=1.25, t1=2.5)
        saved = snapshot([segment])
        segment.zh = 'changed'
        path = self.directory / 'subtitles.srt'
        write_export(path, saved)
        original = path.read_text('utf-8')
        self.assertIn('00:00:01,250 --> 00:00:02,500', original)
        self.assertIn('中文\n日本語', original)
        with patch('jpsub.core.export.os.replace', side_effect=OSError('locked')):
            with self.assertRaises(OSError):
                write_export(path, [segment])
        self.assertEqual(path.read_text('utf-8'), original)


if __name__ == '__main__':
    unittest.main()
