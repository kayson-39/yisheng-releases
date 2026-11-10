import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from jpsub.ui.window import MainWindow


class QueueStatusTests(unittest.TestCase):
    def status(self, state):
        label = Mock()
        window = SimpleNamespace(pipeline=SimpleNamespace(_stop=threading.Event()),
                                 lbl_translate_queue=label,
                                 cfg=SimpleNamespace(get=lambda key, default=None: default))
        MainWindow._on_translation_queue(window, state)
        return label.setText.call_args.args[0]

    def test_idle_does_not_claim_end_to_end_sync(self):
        self.assertEqual(self.status({}), '暂无排队 · 正在聆听')

    def test_asr_backlog_visible_without_translation_queue(self):
        text = self.status({'asr_pending': 3, 'audio_pending': 4})
        self.assertIn('识别中 · 待完成 3 条', text)
        self.assertIn('音频缓冲 4 块', text)

    def test_both_stages_and_retry_count_visible(self):
        text = self.status({'pending':2,'oldest_s':9,'asr_pending':3,'failed':1})
        for fragment in ('最早等待 9s', '识别待完成 3 条', '1 条待补译'):
            self.assertIn(fragment, text)
