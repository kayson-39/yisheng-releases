"""Verify the actual Inno EXE in an isolated install; never install over a user's app."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--installer', type=Path, required=True)
    ap.add_argument('--stage', type=Path, required=True)
    ap.add_argument('--wav', type=Path, required=True)
    ap.add_argument('--native-window', action='store_true',
                    help='briefly show the test window for a native Windows font/render check')
    a = ap.parse_args()
    artifacts = (ROOT / 'artifacts').resolve(strict=True)
    exe, stage = a.installer.resolve(strict=True), a.stage.resolve(strict=True)
    if os.name != 'nt' or not all(p.is_relative_to(artifacts) for p in (exe, stage)):
        raise ValueError('use locally built artifacts on Windows only')
    manifest = (stage / 'manifest.json').read_bytes()
    data = json.loads(manifest)
    version = data['version']
    if exe.name != f'译声实时翻译安装器-{version}.exe':
        raise ValueError('unexpected installer')
    # Retain the log, screenshot and metrics; use a new isolated prefix on every run.
    qa = Path(tempfile.mkdtemp(prefix=f'QA-{version}-', dir=artifacts)).resolve()
    target = qa / '中文安装 测试'
    env = os.environ.copy()
    if target.exists():
        raise FileExistsError(target)
    subprocess.run([str(exe), '/VALIDATE=1', '/CURRENTUSER', '/VERYSILENT', '/SUPPRESSMSGBOXES',
                    '/SP-', '/NORESTART', '/NOCLOSEAPPLICATIONS', '/NOICONS',
                    f'/DIR={target}', f'/LOG={qa / "validation.log"}'], check=True, timeout=900)
    expected = {r['path'] for r in data['files']} | {'manifest.json'}
    actual = {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()}
    if actual != expected or (target / 'manifest.json').read_bytes() != manifest:
        raise ValueError('validation install differs from reviewed stage: ' + str(sorted(actual ^ expected)[:5]))
    for row in data['files']:
        p = target / row['path']
        if p.stat().st_size != row['size'] or digest(p) != row['sha256']:
            raise ValueError('installed content mismatch: ' + row['path'])
    env.update(LOCALAPPDATA=str(qa / '独立用户'), JPSUB_DATA_DIR=str(qa / '独立用户' / 'UserData'),
               JPSUB_MODEL_DIR=str(target / 'app/models'),
               QT_QPA_PLATFORM='windows' if a.native_window else 'offscreen')
    py = target / 'app/runtime/py312/python.exe'
    launcher = target / 'app/scripts/launcher.py'
    def run(args, timeout):
        res = subprocess.run([str(py), '-I', '-S', '-B', '-X', 'utf8', *map(str, args)],
                             cwd=qa, env=env, capture_output=True, encoding='utf-8', errors='replace',
                             timeout=timeout)
        if res.returncode:
            raise RuntimeError(f'{res.returncode}: {(res.stdout + res.stderr)[-2400:]}')
        return res.stdout
    check = run([launcher, '--check'], 90)
    if '可选 GGUF 模型未安装' not in check:
        raise ValueError('unbundled GGUF not reported optional')
    shot = qa / 'first-window.png'
    run([launcher, '--foreground', '--screenshot', shot], 90)
    png = shot.read_bytes()
    if png[:8] != b'\x89PNG\r\n\x1a\n' or min(struct.unpack('>II', png[16:24])) < 100:
        raise ValueError('first window did not render')
    smoke = json.loads(run([ROOT/'tests/smoke_release_audio.py', '--candidate', target,
                            '--wav', a.wav.resolve(strict=True)], 240))
    report = dict(version=version, installer_sha256=digest(exe), installer_bytes=exe.stat().st_size,
                  matched_payload_files=len(data['files']), manifest_verified=True, check=check,
                  screenshot=shot.name, qt_platform=env['QT_QPA_PLATFORM'],
                  audio_smoke=smoke, mode='validation-no-registration')
    (qa / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), 'utf-8')
    print(json.dumps(dict(result=str(qa/'result.json'), **report), ensure_ascii=False))


if __name__ == '__main__':
    main()
