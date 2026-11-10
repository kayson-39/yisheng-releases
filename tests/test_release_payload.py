"""Pure release-policy tests; no host manifest, database or network required."""
import unittest

from tools.release_payload import allow_legacy, validate_path


class ReleasePathTests(unittest.TestCase):
    def test_exact_canonical_path_preserved(self):
        for value in ("译声实时翻译.exe", "app/runtime/py312/Lib/http/client.py",
                      "app/runtime/py312/Scripts/.empty"):
            self.assertEqual(value, validate_path(value))

    def test_unsafe_windows_names_fail_closed(self):
        for value in ("", "/", "/a", "a/", "a//b", "c:/x", "../x", "a/./b", "a/../b",
                      r"a\b", r"\\host\share", "a/b:$DATA", "b.", "b ", "a/nul.txt",
                      "a/NuL .txt", "a/COM¹.txt", "a/lpt2", "a/CONOUT$", "a/SHORT~1/file",
                      "a/$Extend/x", "file?.exe", "a|b", "\x00a", "a\u202eb", "a\ud800",
                      "x/" + "x" * 256, None, 123):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                validate_path(value)
            self.assertFalse(allow_legacy(value))

    def test_known_boot_and_runtime_files(self):
        for path in ("译声实时翻译.exe", "app/runtime/py312/pythonw.exe",
                     "app/runtime/py312/Lib/idlelib/history.py",
                     "app/runtime/packages/httpx/_config.py",
                     "app/runtime/packages/httpx-0.28.1.dist-info/licenses/LICENSE.md",
                     "app/runtime/packages/_cffi_backend.cp312-win_amd64.pyd",
                     "app/runtime/packages/certifi/cacert.pem",
                     "app/engine/cuda/cublasLt64_12.dll",
                     "app/models/parakeet-ja-0.6b-int8/model.int8.onnx",
                     "app/models/argos/en_zh/translate-en_zh-1_9/model/config.json"):
            with self.subTest(path=path):
                self.assertTrue(allow_legacy(path))

    def test_payload_never_recovers_host_or_user_data_by_fallback(self):
        for path in ("app/README.md", "使用说明.md", "unins000.exe", "unins000.dat",
                     "app/scripts/launcher.py", "app/jpsub/main.py", "app/startup.log",
                     "app/work/cache/x", "app/models/private.gguf",
                     "app/models/parakeet-ja-0.6b-int8/test_wavs/test_ja_1.wav",
                     "app/models/llm/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
                     "app/engine/server.log", "app/engine/cpu/private.dll",
                     "app/engine/random/llama-server.exe", "app/engine/cpu/ggml-cuda.dll",
                     "app/runtime/unknown/python.exe", "app/runtime/packages/unknown/plausible.py",
                     "app/runtime/packages/numpy/.env", "app/runtime/packages/numpy/custom.pem",
                     "app/runtime/packages/numpy/private.log.1", "app/runtime/packages/numpy/priv.sqlite",
                     "app/runtime/packages/numpy/credentials/token.txt", "app/runtime/packages/numpy/cache/a",
                     "app/runtime/packages/numpy/__pycache__/mod.pyc",
                     "app/runtime/py312/Lib/site-packages/pip-26.2.1.dist-info/direct_url.json",
                     "config.json", "artifacts/replay.wav", "unknown.txt"):
            with self.subTest(path=path):
                self.assertFalse(allow_legacy(path))
