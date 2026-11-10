"""Read-only audit of the installer and its correspondence to current application sources."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'artifacts/译声 候选-0.10.11'
SETUP = ROOT / 'artifacts/单文件安装器-0.10.11/译声实时翻译安装器-0.10.11.exe'
INSTALLED = Path('C:/Users/pz730/AppData/Local/Programs/YishengTranslatorWhite/app')

def sha(path):
    with path.open('rb') as h:
        return hashlib.file_digest(h, 'sha256').hexdigest()

def main():
    spec = importlib.util.spec_from_file_location('installer_build', ROOT/'packaging/installer_build.py')
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    version, files = build.checked_files(STAGE.absolute())
    assert version == '0.10.11'
    from tools.stage_release import OVERLAYS
    manifest = json.loads((STAGE/'manifest.json').read_text('utf8'))
    index = {r['path']: r for r in manifest['files']}
    compare = dict(OVERLAYS)
    compare.update({p: p[4:] for p in index if p.startswith('app/jpsub/')})
    changed = [p for p, source in compare.items() if sha(ROOT/source) != index[p]['sha256']]
    missing = [p.relative_to(ROOT/'jpsub').as_posix() for p in (ROOT/'jpsub').rglob('*')
               if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts
               and 'app/jpsub/'+p.relative_to(ROOT/'jpsub').as_posix() not in index]
    installed_changed = [p for p in index if p.startswith('app/jpsub/') and
                         (not (INSTALLED/p[4:]).is_file() or sha(INSTALLED/p[4:]) != index[p]['sha256'])]
    actual_sha = sha(SETUP)
    assert not changed and not missing, (changed, missing)
    assert actual_sha == '9735854b5048c75876b0e5fbdbb1af5da2cf7c19e7fb928080e69187eed4ccc1'
    forbidden = [p for p in index if p.lower().endswith(('.gguf', '.gguf.part', '.db', '.sqlite', '.wav', '.log'))
                 or p.lower().endswith('/config.json') or 'userdata/' in p.lower()]
    assert not forbidden, forbidden
    result = dict(version=version, files=len(files), payload_files=len(index),
                  current_sources_checked=len(compare), current_source_changes=changed,
                  new_source_files_missing=missing, installed_source_differences=installed_changed,
                  bytes=SETUP.stat().st_size, sha256=actual_sha,
                  excluded='GGUF, user history/config, API-key files, recordings and logs')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT))
    main()
