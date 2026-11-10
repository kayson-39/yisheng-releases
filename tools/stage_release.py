"""Assemble a LOCAL review candidate, offline. Never archive a whole installation."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import stat
import tempfile
from pathlib import Path, PureWindowsPath

try:
    from .release_payload import allow_legacy, validate_path
except ImportError:  # direct invocation
    from release_payload import allow_legacy, validate_path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {  # same audited binary payload; originals differ in application metadata
    "f2d34648dfe9241a9df0251be6f434deca63a168b0a513eae6469fb8290854b7",
    "70b6b6f72597f04e3a95acd2240ac7a276d89475cc7e8823a91d52e33a55ed55",
}
OVERLAYS = {
    "app/scripts/launcher.py": "packaging/launcher.py",
    "app/jpsub/core/readiness.py": "jpsub/core/readiness.py",
    "app/THIRD_PARTY_NOTICES.md": "THIRD_PARTY_NOTICES.md",
    "app/LICENSE": "LICENSE", "app/requirements.txt": "requirements.txt",
    "app/README.md": "packaging/使用说明.md", "使用说明.md": "packaging/使用说明.md",
    "app/models/llm/LICENSE-Sakura-GalTransl-7B.txt": "packaging/Sakura-NOTICE.txt",
}


def version():
    module = ast.parse((ROOT / "jpsub/__init__.py").read_text("utf-8"))
    return next(ast.literal_eval(n.value) for n in module.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "__version__")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def regular(root, rel):
    name = PureWindowsPath(validate_path(rel))
    if name.anchor or any(x in ("", ".", "..") or ":" in x for x in rel.split("/")) or "\\" in rel:
        raise ValueError(f"invalid path: {rel}")
    path = root.joinpath(*rel.split("/"))
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing or out of bounds: {rel}")
    for p in (path, *path.parents):
        if p == root:
            break
        if getattr(p.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or p.is_symlink():
            raise ValueError(f"linked input: {rel}")
    return path


def plan(original):
    manifest = regular(original, "发行文件清单.json").read_bytes()
    if hashlib.sha256(manifest).hexdigest() not in BASELINES:
        raise ValueError("旧文件清单不是已审核基线，拒绝构建。")
    entries = {}
    for row in json.loads(manifest)["files"]:
        rel = row["path"]
        if rel in OVERLAYS:
            continue
        if rel.startswith("app/jpsub/"):  # exact names from the pinned manifest only
            entries[rel] = (regular(ROOT, rel[4:]), None)
        elif allow_legacy(rel) and not rel.startswith("app/scripts/"):
            entries[rel] = (regular(original, rel), row["sha256"].lower())
    entries.update({k: (regular(ROOT, v), None) for k, v in OVERLAYS.items()})
    if len({k.casefold() for k in entries}) != len(entries) or any(k.lower().endswith(".gguf") for k in entries):
        raise ValueError("目标文件名冲突或误收录了另行下载的大模型。")
    for name in ("译声实时翻译.exe", "app/runtime/py312/pythonw.exe", "app/runtime/py312/python.exe",
                 "app/engine/cpu/llama-server.exe", "app/engine/vulkan/llama-server.exe",
                 "app/engine/cuda/llama-server.exe", "app/models/silero_vad.onnx",
                 "app/models/parakeet-ja-0.6b-int8/model.int8.onnx", "app/scripts/launcher.py"):
        if name not in entries:
            raise ValueError("missing required input: " + name)
    return entries


def materialize(entries, out):
    out = out.absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"请使用新的输出目录：{out}")
    size = sum(path.stat().st_size for path, _ in entries.values())
    if shutil.disk_usage(out.parent).free < size + (32 << 20):
        raise OSError("磁盘剩余空间不足")
    # TemporaryDirectory cleans only the private directory created by this operation.
    with tempfile.TemporaryDirectory(prefix=".candidate-", dir=out.parent) as tmp:
        payload = Path(tmp) / "payload"
        payload.mkdir()
        files = []
        for rel, (source, expected) in sorted(entries.items()):
            target = payload / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            actual = digest(target)
            if expected and actual != expected:
                raise ValueError(f"内容不匹配，未生成候选目录：{rel}")
            files.append(dict(path=rel, size=target.stat().st_size, sha256=actual))
        (payload / "manifest.json").write_text(json.dumps(dict(schema=1, version=version(), files=files),
                                              ensure_ascii=False, indent=1), encoding="utf-8")
        if out.exists():
            raise FileExistsError(out)
        payload.rename(out)  # Windows: fails rather than replacing an existing target
    return len(files), size


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True, help="已审核的二进制来源根目录")
    ap.add_argument("--output", type=Path, help="新目录；不指定时只检查输入清单和空间需求")
    args = ap.parse_args()
    base = args.base.resolve(strict=True)
    if args.output and (args.output.resolve().is_relative_to(base)
                        or not args.output.parent.is_dir()):
        ap.error("输出须位于已存在的父目录，且不得在来源目录内")
    entries = plan(base)
    if args.output:
        count, size = materialize(entries, args.output)
        print(f"Verified local candidate: {count} files, {size} bytes -> {args.output}")
    else:
        print(f"Plan only (contents not yet verified): {len(entries)} files, "
              f"{sum(p.stat().st_size for p, _ in entries.values())} bytes")


if __name__ == "__main__":
    main()
