"""Privacy/Windows-path boundary for importing an **authenticated** legacy payload.

This is NOT a directory walker. Only apply allow_legacy to members of the
fingerprint-pinned release manifest, verifying each file's length and SHA-256.
Do not trust filenames as provenance. Callers must also reject case-insensitive
duplicates and filesystem reparse points in source and destination ancestry.
"""
from __future__ import annotations

import re
import unicodedata


def validate_path(path: str) -> str:
    """Accept a canonical archive member, not a Windows path or a directory."""
    if not isinstance(path, str) or not path or unicodedata.normalize("NFC", path) != path:
        raise ValueError("invalid archive path")
    if any(c in '<>:"\\|?*' or unicodedata.category(c).startswith("C") for c in path):
        raise ValueError("forbidden Windows path character")
    for part in path.split("/"):
        stem = part.split(".", 1)[0].rstrip(" ").upper()
        if (not part or part in {".", ".."} or part.endswith((" ", "."))
                or len(part.encode("utf-16-le")) > 510 or part.startswith("$")
                or re.search(r"~[0-9]", part)
                or stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"}
                or re.fullmatch(r"(COM|LPT)[0-9¹²³]", stem)):
            raise ValueError("unsafe or ambiguous Windows path component")
    return path


# Exact individual legacy root metadata allowed; never import app code,
# launch scripts or the working README from this machine through this policy.
_ROOT = {
    "译声实时翻译.exe", "安装向导中文翻译许可.txt", "app/LICENSE", "app/THIRD_PARTY_NOTICES.md",
    "app/requirements.txt",
}
_MODELS = {
    "silero_vad.onnx", "parakeet-ja-0.6b-int8/model.int8.onnx", "parakeet-ja-0.6b-int8/tokens.txt",
    "llm/LICENSE-Sakura-GalTransl-7B.txt", "llm/Sakura-Galtransl-7B-v3.7-IQ4_XS.source.json",
    "argos/index.json",
}
for _prefix, _extras in (
    ("argos/ja_en/ja_en", ["model/shared_vocabulary.txt", "stanza/ja/tokenize/gsd.pt"]),
    ("argos/en_zh/translate-en_zh-1_9", ["model/config.json", "model/shared_vocabulary.json",
                                         "stanza/en/tokenize/ewt.pt"]),
):
    _MODELS.update(f"{_prefix}/{name}" for name in
                   ["metadata.json", "model/model.bin", "README.md", "sentencepiece.model",
                    "stanza/resources.json", *_extras])

_ENGINE_SHARED = set("""
concrt140.dll ggml-base.dll ggml-rpc.dll ggml.dll libomp.dll LICENSE-LLVM-OpenMP
llama-batched-bench-impl.dll llama-bench-impl.dll llama-cli-impl.dll
llama-common.dll llama-completion-impl.dll llama-fit-params-impl.dll
llama-perplexity-impl.dll llama-quantize-impl.dll llama-server-impl.dll
llama-server.exe llama.dll msvcp140_1.dll msvcp140_2.dll msvcp140_codecvt_ids.dll
msvcp140.dll mtmd.dll vcomp140.dll vcruntime140_1.dll vcruntime140.dll
""".split())
_ENGINE_SHARED.update(f"ggml-cpu-{name}.dll" for name in """
alderlake cannonlake cascadelake cooperlake haswell icelake ivybridge piledriver
sandybridge sapphirerapids skylakex sse42 x64 zen4
""".split())
_ENGINE = {
    "cpu": _ENGINE_SHARED,
    "vulkan": _ENGINE_SHARED | {"ggml-vulkan.dll"},
    "cuda": _ENGINE_SHARED | {"ggml-cuda.dll", "cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"},
}

# Package classes are a second boundary below the pinned manifest. New packages
# and runtimes require a conscious policy change, not a wider directory glob.
_PACKAGES = set("""
_yaml aiohappyeyeballs aiohttp aiosignal anyio attr attrs av av.libs certifi cffi
click colorama ctranslate2 edge_playback edge_tts faster_whisper filelock flatbuffers
frozenlist fsspec google h11 hf_xet httpcore httpcore2 httpx httpx2 huggingface_hub
idna multidict numpy numpy.libs onnxruntime packaging PIL pip propcache psutil pycparser
PySide6 sentencepiece sherpa_onnx shiboken6 soundcard soxr tabulate tokenizers tqdm
truststore yaml yarl
""".split())
_DIST_NAMES = {s.lower() for s in _PACKAGES} | {
    "protobuf", "pillow", "pyside6_essentials", "pyyaml", "typing_extensions", "sherpa_onnx_core",
}
_PY_DIRS = {"DLLs", "Lib", "include", "libs", "tcl"}
_PY_FILES = set("""
BUILD concrt140.dll LICENSE.txt msvcp140_1.dll msvcp140_2.dll msvcp140_codecvt_ids.dll
msvcp140.dll python.exe python3.dll python312.dll pythonw.exe vcomp140.dll
vcruntime140_1.dll vcruntime140.dll
""".split())
_PRIVATE_PARTS = {
    "work", "userdata", "user_data", "cache", "logs", "history", "backups", "migration-backups",
    "credentials", "secrets", ".git", ".svn", ".hg", "__pycache__",
}
_PRIVATE_FILES = {
    "config.json", "config.json.bak", "settings.json", "direct_url.json", "pyvenv.cfg",
    ".netrc", "_netrc", "id_rsa", "id_ed25519", "token", "credentials.json", "api_key",
}
_PRIVATE_SUFFIXES = (
    ".bak", ".backup", ".tmp", ".part", ".log", ".db", ".db-wal", ".db-shm", ".sqlite",
    ".sqlite3", ".sqlite-wal", ".sqlite-shm", ".pyc", ".pyo", ".key", ".pfx", ".p12",
    ".wav", ".mp3", ".mp4", ".srt", ".vtt",
)
_CERTIFICATES = {
    "app/runtime/packages/certifi/cacert.pem",
    "app/runtime/packages/pip/_vendor/certifi/cacert.pem",
    "app/runtime/py312/Lib/site-packages/pip/_vendor/certifi/cacert.pem",
}


def allow_legacy(path: str) -> bool:
    """Would copying this *pinned legacy manifest member* be in scope?"""
    try:
        validate_path(path)
    except ValueError:
        return False
    parts = path.split("/")
    if path in _ROOT or path == "app/engine/runtime_profiles.json":
        return True
    if parts[:2] == ["app", "models"]:
        # A runtime model's exact config.json is public; app/user config is not.
        return "/".join(parts[2:]) in _MODELS
    if parts[:2] == ["app", "engine"] and len(parts) == 4:
        return parts[3] in _ENGINE.get(parts[2], ())
    if parts[:2] != ["app", "runtime"] or len(parts) < 4:
        return False
    lowered = [p.lower() for p in parts]
    name = lowered[-1]
    if (_PRIVATE_PARTS.intersection(lowered) or name in _PRIVATE_FILES
            or name.endswith(_PRIVATE_SUFFIXES) or ".log." in name
            or any(p.startswith(".") and p not in {".f2py_f2cmap", ".empty"} for p in parts)
            or (name.endswith(".pem") and path not in _CERTIFICATES)):
        return False
    if parts[2] == "py312":
        return ((len(parts) == 4 and parts[3] in _PY_FILES)
                or (len(parts) >= 5 and parts[3] in _PY_DIRS)
                or path == "app/runtime/py312/Scripts/.empty")
    if parts[2] == "packages":
        if len(parts) == 4:
            return parts[3] in {"typing_extensions.py", "_cffi_backend.cp312-win_amd64.pyd"}
        pkg = parts[3]
        if pkg in _PACKAGES:
            return True
        match = re.fullmatch(r"([a-z0-9_]+)-[0-9][A-Za-z0-9.+]*\.dist-info", pkg)
        return bool(match and match[1] in _DIST_NAMES)
    return False
