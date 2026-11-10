"""Offline first-use checks: isolated models, no GUI, devices, or native inference."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jpsub import main, paths
from jpsub.core import asr, config, llama_engine, local_mt
from jpsub.ui.window import MainWindow


class FirstUseTests(unittest.TestCase):
    def setUp(self):
        stack = self.enterContext(contextlib.ExitStack())
        self.root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        self.models = self.root / "模型"
        self.jp = self.models / "parakeet-ja-0.6b-int8"
        self.jp.mkdir(parents=True)
        stack.enter_context(patch.object(paths, "MODELS", self.models))
        stack.enter_context(patch.object(asr, "MODEL_DIR", self.jp))
        stack.enter_context(patch.object(llama_engine, "MODELS_LLM", self.models / "llm"))
        stack.enter_context(patch.object(llama_engine, "ROOT", self.root))
        stack.enter_context(patch.object(llama_engine, "ENGINE_DIR", self.root / "engine"))
        stack.enter_context(patch.object(local_mt, "ARGOS_ROOT", self.models / "argos"))
        self.cfg = config.Config(self.root / "user.json")

    def check(self, *, installation=False):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = main._selftest(self.cfg, installation=installation)
        return result, out.getvalue()

    def speech_files(self):
        (self.jp / "model.int8.onnx").write_bytes(bytes(1024 * 1024))
        (self.jp / "tokens.txt").write_text("あ 1\n", encoding="utf-8")
        (self.models / "silero_vad.onnx").write_bytes(bytes(101_000))

    def test_empty_placeholders_cannot_pass_asr_and_vad_check(self):
        for f in (self.jp/"model.int8.onnx", self.jp/"tokens.txt", self.models/"silero_vad.onnx"):
            f.touch()
        self.cfg.set("translate.enabled", False, save=False)
        self.assertFalse(asr.FinalASR.available())
        self.assertNotEqual(self.check()[0], 0)

    def test_missing_default_translation_assets_fail_despite_complete_speech_assets(self):
        self.speech_files()
        result, output = self.check()
        self.assertNotEqual(result, 0)
        self.assertIn("GGUF", output)
        self.assertIn("引擎", output)

    def test_cloud_or_disabled_translation_needs_no_local_llm_or_network(self):
        self.speech_files()
        self.cfg.set("translate.provider", "ms-free", save=False)
        with patch("socket.socket.connect", side_effect=AssertionError("unexpected network")):
            self.assertEqual(self.check()[0], 0)
        self.cfg.set("translate.enabled", False, save=False)
        self.cfg.set("translate.provider", "llamacpp", save=False)
        self.assertEqual(self.check()[0], 0)

    def test_localmt_without_weights_and_tokenizer_fails(self):
        self.speech_files()
        self.cfg.set("translate.provider", "localmt", save=False)
        direct = local_mt.ARGOS_ROOT / "ja_zh"
        (direct / "model").mkdir(parents=True)
        (direct / "sentencepiece.model").touch()
        self.assertNotEqual(self.check()[0], 0)
        (direct / "model" / "model.bin").write_bytes(b"weights")
        (direct / "sentencepiece.model").write_bytes(b"spm")
        self.assertEqual(self.check()[0], 0)

    def test_ensure_dirs_only_creates_user_owned_paths(self):
        user = self.root / "新用户"
        with patch.multiple(paths, USER_DATA=user, CACHE=user/"cache", HISTORY=user/"history",
                            MODELS=self.root/"missing-models", BACKGROUNDS=self.root/"missing-background"):
            paths.ensure_dirs()
            self.assertTrue(paths.HISTORY.is_dir())
            self.assertFalse(paths.MODELS.exists())
            self.assertFalse(paths.BACKGROUNDS.exists())

    def bundled(self):
        self.speech_files()
        cpu = llama_engine.ENGINE_DIR / "cpu"
        cpu.mkdir(parents=True)
        for name in ("llama-server.exe", "llama-server-impl.dll", "llama.dll", "ggml.dll",
                     "ggml-base.dll", "ggml-cpu-any.dll"):
            (cpu / name).write_bytes(b"fixture")
        fallback = local_mt.ARGOS_ROOT / "ja_zh"
        (fallback/"model").mkdir(parents=True)
        (fallback/"model"/"model.bin").write_bytes(b"fixture")
        (fallback/"sentencepiece.model").write_bytes(b"fixture")
        llama_engine.MODELS_LLM.mkdir()
        return llama_engine.MODELS_LLM / llama_engine.RECOMMENDED_MODELS[0]["name"]

    def test_installed_components_pass_without_optional_gguf_but_strict_readiness_does_not(self):
        self.bundled()
        result, output = self.check(installation=True)
        self.assertEqual(result, 0)
        self.assertIn("可选 GGUF 模型未安装", output)
        self.assertNotEqual(self.check()[0], 0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(["--check", "--config", str(self.cfg.path)]), 0)
        self.assertFalse(self.cfg.path.exists())

    def test_corrupt_optional_gguf_differs_from_absent_and_existing_valid_is_reused(self):
        gguf = self.bundled()
        gguf.touch()
        self.assertNotEqual(self.check(installation=True)[0], 0)
        gguf.write_bytes(b"GGUF" + bytes(1024*1024))
        self.assertEqual(self.check(installation=True)[0], 0)
        self.assertEqual(self.check()[0], 0)

    def test_explicit_live_provider_checked_only_when_active(self):
        self.speech_files()
        self.cfg.set("translate.provider", "ms-free", save=False)
        self.cfg.set("translate.live_provider", "llamacpp", save=False)
        self.assertNotEqual(self.check()[0], 0)
        self.cfg.set("asr.draft_mode", "off", save=False)
        self.assertEqual(self.check()[0], 0)

    def test_missing_model_explains_choices_without_changing_provider_or_downloading(self):
        from unittest.mock import Mock
        label = Mock()
        with patch.object(llama_engine, "download_model", side_effect=AssertionError("download")):
            MainWindow._update_trans_label_from_cfg(SimpleNamespace(cfg=self.cfg, lbl_trans=label))
        self.assertIn("轻量翻译", label.setText.call_args.args[0])
        self.assertIn("GGUF", label.setToolTip.call_args.args[0])
        self.assertEqual(self.cfg.get("translate.provider"), "llamacpp")

    def test_main_selftest_does_not_migrate_or_write_user_directories(self):
        with patch.object(main, "ensure_dirs", side_effect=AssertionError("unexpected write")), \
                patch.object(config.Config, "_migrate_legacy", side_effect=AssertionError("unexpected migration")), \
                patch.object(config, "CONFIG_PATH", self.root/"new.json"), \
                patch.object(main, "CONFIG_PATH", self.root/"new.json"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main.main(["--selftest"]), 1)
        self.assertFalse((self.root/"new.json").exists())
