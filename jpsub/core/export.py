"""Export immutable snapshots; never touch Qt widgets from the writer thread."""
from __future__ import annotations
import os, tempfile
from pathlib import Path
from dataclasses import replace

def snapshot(segments):
    return tuple(replace(s) for s in segments)

def to_srt(segments):
    def fmt(t):
        n=max(0,round(t*1000))
        return f'{n//3600000:02}:{n//60000%60:02}:{n//1000%60:02},{n%1000:03}'
    return '\n\n'.join(f'{i}\n{fmt(s.t0)} --> {fmt(max(s.t1,s.t0+.5))}\n'+ '\n'.join(x for x in (s.zh,s.jp) if x)
        for i,s in enumerate(segments,1))+'\n'

def write_export(path, segments):
    path=Path(path)
    text=('\n\n'.join('\n'.join(x for x in (s.jp,s.zh) if x) for s in segments)+'\n') if path.suffix.lower()=='.txt' else to_srt(segments)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as f:
            temporary=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if temporary and temporary.exists():temporary.unlink()
