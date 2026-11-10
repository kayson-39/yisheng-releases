"""Validate a staged release and build with the verified legacy Inno Setup 7 compiler."""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
import os
import re
import stat
import subprocess
import tempfile


REQUIRED = {
    '译声实时翻译.exe', 'app/jpsub/__init__.py', 'app/scripts/launcher.py',
    'app/runtime/py312/pythonw.exe', 'app/runtime/py312/python.exe',
}
MAX_SETUP_BYTES = 2**31 - 1  # conservative PE/Authenticode compatibility limit


def regular(path):
    mode = path.lstat()
    if not stat.S_ISREG(mode.st_mode) or getattr(mode, 'st_file_attributes', 0) & 0x400:
        raise ValueError(f'不是普通文件或为重解析点: {path}')


def safe_rel(s):
    if not isinstance(s, str) or not s or re.search(r'[\\:<>"|?*{}\x00-\x1f]', s):
        raise ValueError(f'不安全的清单路径: {s!r}')
    for item in s.split('/'):
        if item in ('', '.', '..') or item.endswith((' ', '.')) or PureWindowsPath(item).is_reserved():
            raise ValueError(f'不安全的清单路径: {s!r}')
    return s


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_files(stage):
    stage_info = stage.lstat()
    if (not stat.S_ISDIR(stage_info.st_mode) or getattr(stage_info, 'st_file_attributes', 0) & 0x400
            or stage.resolve() != stage.absolute()):
        raise ValueError('stage 目录不可通过重解析点或符号链接访问')
    owned = {}
    for directory, dirs, files in os.walk(stage, onerror=lambda e: (_ for _ in ()).throw(e)):
        base = Path(directory)
        for entry in dirs:
            info = (base / entry).lstat()
            if not stat.S_ISDIR(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError(f'不接受目录重解析点: {entry}')
        for entry in files:
            p = base / entry
            regular(p)
            rel = safe_rel(p.relative_to(stage).as_posix())
            if rel.casefold() in owned:
                raise ValueError(f'文件名在 Windows 上冲突: {rel}')
            owned[rel.casefold()] = rel
    manifest = json.loads((stage / 'manifest.json').read_text('utf-8-sig'))
    version = manifest.get('version', '')
    if (manifest.get('schema') != 1 or not isinstance(manifest.get('files'), list)
            or not isinstance(version, str) or not re.fullmatch(r'(?:0|[1-9]\d{0,3})(?:\.(?:0|[1-9]\d{0,3})){2}', version)):
        raise ValueError('不支持的发布清单格式或版本')
    seen = {'manifest.json'}
    for row in manifest['files']:
        rel = safe_rel(row['path'])
        if rel.casefold().endswith(('.gguf', '.gguf.part', '.gguf.tmp')):
            raise ValueError(f'单文件发行包禁止内置 GGUF 权重: {rel}')
        if rel.casefold() in seen or owned.get(rel.casefold()) != rel:
            raise ValueError(f'清单文件缺失、重复或大小写不匹配: {rel}')
        p = stage / rel
        if (type(row['size']) is not int or row['size'] != p.stat().st_size
                or not isinstance(row['sha256'], str) or digest(p) != row['sha256'].lower()):
            raise ValueError(f'文件校验失败: {rel}')
        seen.add(rel.casefold())
    if set(owned) != seen or not REQUIRED.issubset(owned.values()):
        raise ValueError('stage 存在未登记的文件，或缺少 Windows 启动所需文件')
    code = (stage / 'app/jpsub/__init__.py').read_text('utf-8-sig')
    if re.search(r'^__version__\s*=\s*"' + re.escape(version) + r'"\s*$', code, re.M) is None:
        raise ValueError('应用版本与发布清单不一致')
    return version, sorted(owned.values())


def compile_setup(stage, output, iscc):
    stage, output, iscc = Path(stage).absolute(), Path(output).resolve(), Path(iscc).resolve()
    if not iscc.is_file() or iscc.name.casefold() != 'iscc.exe':
        raise ValueError('请指定可信 Inno Setup 7 的 ISCC.exe')
    if output == stage or output.is_relative_to(stage) or output.exists():
        raise ValueError('输出必须是 stage 外的新目录，不能覆盖已有产物')
    if any(c in str(stage) for c in '"{}\r\n'):
        raise ValueError('stage 根路径不能含引号、花括号或换行')
    version, paths = checked_files(stage)
    with tempfile.TemporaryDirectory(prefix='yisheng-setup-') as temp:
        build = Path(temp)
        checked = build / 'files.iss'
        lines = []
        for rel in paths:
            parent = str(PureWindowsPath(rel).parent)
            dest = '{app}' if parent == '.' else '{app}\\' + parent
            # Ship and repair the exact required ASR/VAD files. Never enumerate or
            # clean the destination: imported GGUF and other unlisted files stay put.
            flags = 'ignoreversion uninsneveruninstall' if rel.casefold().startswith('app/models/') else 'ignoreversion'
            lines.append(f'Source: "{stage / rel}"; DestDir: "{dest}"; Flags: {flags}')
        checked.write_text('\n'.join(lines), encoding='utf-8-sig')
        # Unicode BOM also works when the Windows ANSI code page is not Chinese.
        script = build / 'installer.iss'
        script.write_text(Path(__file__).with_name('installer.iss').read_text('utf-8-sig'),
                          encoding='utf-8-sig')
        output.mkdir(exist_ok=False)
        subprocess.run([str(iscc), f'/DBuildOut={output}', f'/DProductVersion={version}', f'/DStageRoot={stage}',
                        f'/DCheckedFiles={checked}', str(script)], check=True)
        verify_output(output, version)
        return version


def verify_output(output, version):
    setup = output / f'译声实时翻译安装器-{version}.exe'
    if list(output.iterdir()) != [setup]:
        raise RuntimeError('必须恰好生成一个安装 EXE；含分卷、额外文件或缺少 EXE 的目录不是产物')
    regular(setup)
    if not 0 < setup.stat().st_size <= MAX_SETUP_BYTES:
        raise RuntimeError('安装 EXE 为空或超过单文件发布上限，请减小负载，不可改用分卷')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, type=Path)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--iscc', type=Path)
    a = p.parse_args()
    if a.check_only:
        v, files = checked_files(a.stage.absolute())
        print(f'OK {v}: {len(files)} files (including manifest)')
    elif not a.output or not a.iscc:
        p.error('compile requires --output and --iscc')
    else:
        print(f'Built {compile_setup(a.stage, a.output, a.iscc)}: {a.output}')


if __name__ == '__main__':
    main()
