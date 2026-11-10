import unittest
from unittest.mock import patch
from jpsub.core import llama_engine as engine


class OfflineEngineTests(unittest.TestCase):
    def test_missing_engine_never_downloads_or_extracts_binaries(self):
        with patch.object(engine, 'installed', return_value=False), patch.object(engine.httpx, 'stream') as net:
            for variant in engine.VARIANTS:
                with self.subTest(variant=variant), self.assertRaisesRegex(RuntimeError, '完整发行包'):
                    engine.download_engine(variant)
            net.assert_not_called()

    def test_present_engine_can_be_reused_without_network(self):
        with patch.object(engine, 'installed', return_value=True), patch.object(engine.httpx, 'stream') as net:
            self.assertEqual(engine.download_engine('cpu'), engine.engine_dir('cpu'))
            net.assert_not_called()
