import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import stage_release as stage
from jpsub.core import model_catalog


class StagingTests(unittest.TestCase):
    def test_rejects_replaced_legacy_manifest_before_reading_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp).resolve()
            (p / "发行文件清单.json").write_text('{"files":[]}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, '基线'):
                stage.plan(p)

    def test_mismatch_exposes_no_candidate_and_existing_dir_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            src, out = p / "input", p / "candidate"
            src.write_bytes(b"a")
            with self.assertRaisesRegex(ValueError, '不匹配'):
                stage.materialize({'app/input': (src, 'bad')}, out)
            self.assertFalse(out.exists())
            entries = {'app/input': (src, hashlib.sha256(b"a").hexdigest())}
            self.assertEqual((1, 1), stage.materialize(entries, out))
            with self.assertRaises(FileExistsError):
                stage.materialize(entries, out)

    def test_public_model_never_uses_a_cached_account_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(model_catalog, 'MODELS_DIR', Path(tmp)), \
                patch('huggingface_hub.snapshot_download', side_effect=RuntimeError('offline fixture')) as get:
            with self.assertRaisesRegex(RuntimeError, 'offline fixture'):
                model_catalog.download('parakeet-ja')
            self.assertIs(get.call_args.kwargs.get('token'), False)
