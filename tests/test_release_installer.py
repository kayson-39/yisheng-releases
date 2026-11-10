"""No compiler, network, registry, audio, or actual install is used by these tests."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1] / 'packaging'
SPEC = importlib.util.spec_from_file_location('installer_build', PROJECT / 'installer_build.py')
ib = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ib)


class InstallerBuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.stage = self.work / '输出 测试'
        self.stage.mkdir()
        self.rows = []
        for name in sorted(ib.REQUIRED):
            p = self.stage / name
            p.parent.mkdir(exist_ok=True, parents=True)
            data = b'__version__ = "0.10.11"\n' if name.endswith('.py') else b'fixture only'
            p.write_bytes(data)
            self.rows.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
        self.manifest()
        self.compiler = self.work / 'ISCC.exe'
        self.compiler.touch()

    def manifest(self, version='0.10.11'):
        (self.stage / 'manifest.json').write_text(json.dumps(dict(schema=1, version=version, files=self.rows)),
                                                  encoding='utf-8')

    def test_checked_unicode_stage_matches_exact_inventory(self):
        version, paths = ib.checked_files(self.stage)
        self.assertEqual(version, '0.10.11')
        self.assertEqual(set(paths), ib.REQUIRED | {'manifest.json'})

    def test_unknown_or_changed_file_stops_build(self):
        unknown = self.stage / 'history.db'
        unknown.touch()
        with self.assertRaises(ValueError):
            ib.checked_files(self.stage)
        unknown.unlink()
        (self.stage / self.rows[0]['path']).write_bytes(b'x' * self.rows[0]['size'])
        with self.assertRaises(ValueError):
            ib.checked_files(self.stage)

    def test_reject_windows_escape_ads_and_case_duplicates(self):
        for name in ('../secret', 'a//b', 'D:/a', '/outside', 'a\\..\\x', 'aux.txt', 'a:b', 'trailing.'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ib.safe_rel(name)
        self.rows.append(dict(self.rows[0], path=self.rows[0]['path'].swapcase()))
        self.manifest()
        with self.assertRaises(ValueError):
            ib.checked_files(self.stage)

    def test_reject_source_version_mismatch(self):
        self.manifest(version='0.10.12')
        with self.assertRaises(ValueError):
            ib.checked_files(self.stage)

    def test_missing_required_bootstrap_blocks_build(self):
        row = next(r for r in self.rows if r['path'].endswith('/launcher.py'))
        (self.stage / row['path']).unlink()
        self.rows.remove(row)
        self.manifest()
        with self.assertRaises(ValueError):
            ib.checked_files(self.stage)

    def test_output_must_be_new_and_external(self):
        with patch.object(subprocess, 'run') as compiler:
            for path in (self.work, self.stage / 'nested'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    ib.compile_setup(self.stage, path, self.compiler)
            compiler.assert_not_called()

    def test_compiler_gets_bom_and_explicit_checked_files_not_globs(self):
        output = self.work / '候选 001'
        model = 'app/models/silero_vad.onnx'
        (self.stage / model).parent.mkdir(parents=True)
        (self.stage / model).write_bytes(b'fixture')
        self.rows.append(dict(path=model, size=7, sha256=hashlib.sha256(b'fixture').hexdigest()))
        self.manifest()
        def fake_compile(args, check):
            self.assertTrue(check)
            self.assertEqual(args[0], str(self.compiler))
            checked = Path(next(a.partition('=')[2] for a in args if a.startswith('/DCheckedFiles=')))
            lines = checked.read_text('utf-8-sig').splitlines()
            self.assertEqual(len(lines), len(self.rows) + 1)
            self.assertTrue(all('*' not in l for l in lines))
            self.assertTrue(any('译声实时翻译.exe' in l for l in lines))
            vad = next(l for l in lines if 'silero_vad.onnx' in l)
            self.assertIn('ignoreversion uninsneveruninstall', vad)
            self.assertNotIn('onlyifdoesntexist', vad)
            self.assertTrue(Path(args[-1]).read_bytes().startswith(b'\xef\xbb\xbf'))
            (output / '译声实时翻译安装器-0.10.11.exe').write_bytes(b'MZfixture')
        with patch.object(subprocess, 'run', side_effect=fake_compile):
            self.assertEqual(ib.compile_setup(self.stage, output, self.compiler), '0.10.11')

    def test_never_target_user_gguf_for_replacement(self):
        payload = b'GGUFfixture'
        weight = 'app/models/llm/large.gguf'
        (self.stage / weight).parent.mkdir(parents=True)
        (self.stage / weight).write_bytes(payload)
        self.rows.append(dict(path=weight, size=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
        self.manifest()
        with self.assertRaisesRegex(ValueError, 'GGUF'):
            ib.checked_files(self.stage)

    def test_block_spanned_or_oversize_output(self):
        out = self.work / 'dist'
        out.mkdir()
        exe = out / '译声实时翻译安装器-0.10.11.exe'
        exe.write_bytes(b'EXE')
        extra = out / 'setup-1.bin'
        extra.touch()
        with self.assertRaises(RuntimeError):
            ib.verify_output(out, '0.10.11')
        extra.unlink()
        with patch.object(ib, 'MAX_SETUP_BYTES', 2), self.assertRaises(RuntimeError):
            ib.verify_output(out, '0.10.11')

    def test_installer_policy_is_per_user_single_exe_no_data_deletion(self):
        script = (PROJECT / 'installer.iss').read_text('utf-8')
        for required in ('PrivilegesRequired=lowest', 'DiskSpanning=no',
                         'Local\\yisheng_white_release_single_instance', 'compiler:Languages\\ChineseSimplified.isl',
                         'Uninstallable=not ValidationInstall', 'CreateUninstallRegKey=not ValidationInstall',
                         'not FileExists(TargetModel)', 'Hash: "{#ModelHash}"', 'Check: not ValidationInstall',
                         'ArchitecturesInstallIn64BitMode=x64compatible',
                         '977B6A85-A0B2-4F52-8915-A388F2A7F289', 'ComparePackedVersion('):
            self.assertIn(required, script)
        self.assertNotIn('[UninstallDelete]', script)
        self.assertNotIn('[InstallDelete]', script)
        self.assertNotIn('UserData', script)

    @unittest.skipUnless(os.environ.get('YISHENG_TEST_ISCC'), 'optional integration: compiler explicitly selected')
    def test_real_compiler_accepts_fixture_without_installing(self):
        for name, data in [('app/jpsub/assets/icon.ico', (PROJECT.parent/'jpsub/assets/icon.ico').read_bytes()),
                           ('使用说明.md', '仅用于安装编译器测试。'.encode('utf-8'))]:
            p = self.stage / name
            p.parent.mkdir(exist_ok=True, parents=True)
            p.write_bytes(data)
            self.rows.append(dict(path=name, size=len(data), sha256=hashlib.sha256(data).hexdigest()))
        self.manifest()
        self.assertEqual(ib.compile_setup(self.stage, self.work / '实际编译',
                                          os.environ['YISHENG_TEST_ISCC']), '0.10.11')
