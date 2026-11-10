# -*- coding: utf-8 -*-
"""内置本地大模型引擎（llama.cpp / llama-server）：下载、启动、停止、健康检查。

不需要用户安装 Ollama；引擎随软件分发（约 47MB），模型使用 GGUF 文件。
"""
from __future__ import annotations

import atexit
import ctypes
import hashlib
import os
import socket
import subprocess
import threading
import time

from pathlib import Path
from typing import Callable, Optional

import httpx

try:                     # psutil 在 requirements.txt 里；万一环境装漏了也不该连引擎都用不了
    import psutil
except Exception:  # noqa: BLE001
    psutil = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = ROOT / "engine"
from ..paths import MODELS, USER_DATA
MODELS_LLM = MODELS / "llm"
SERVER_LOG = USER_DATA / "llama-server.log"

VARIANTS = ("cpu", "vulkan", "cuda")
GPU_VARIANTS = ("vulkan", "cuda")     # 这些变体要用显卡
ProgressCb = Optional[Callable[[str, int, int], None]]

MODEL_MIRRORS = ("https://hf-mirror.com", "https://huggingface.co")
RECOMMENDED_MODELS = [
    dict(
        name="Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf",
        repo="SakuraLLM/Sakura-GalTransl-7B-v3.7",
        revision="759d76b1745f4428308de6564c5ab710d4358b2e",
        desc="Sakura-GalTransl 7B（直播口语 · 6GB显存配置）",
        size_gb=4.25,
        sha256="8f515bf4769f279a7fcf43e57446455a9d4de7f65b1bc9eddee76717e1ff7919",
    ),
]


def download_model(item: dict, on_progress: ProgressCb = None) -> Path:
    """下载推荐 GGUF 模型（hf-mirror 国内镜像优先，支持断点续传）。"""
    filename = str(item.get("name") or "")
    repo = str(item.get("repo") or "")
    if not filename or not repo:
        raise RuntimeError("模型配置不完整")
    MODELS_LLM.mkdir(parents=True, exist_ok=True)
    dest = MODELS_LLM / filename
    if valid_model(dest) and _matches_checksum(dest, item.get("sha256")):
        return dest
    part = MODELS_LLM / (filename + ".part")
    timeout = httpx.Timeout(connect=20.0, read=45.0, write=20.0, pool=20.0)
    last_err: Exception | None = None
    for base in MODEL_MIRRORS:
        revision = str(item.get("revision") or "main")
        url = f"{base}/{repo}/resolve/{revision}/{filename}"
        for _attempt in range(4):
            try:
                already = part.stat().st_size if part.exists() else 0
                headers = {"Range": f"bytes={already}-"} if already else {}
                with httpx.stream("GET", url, headers=headers, timeout=timeout,
                                  follow_redirects=True) as r:
                    if r.status_code == 416 and already:
                        part.unlink(missing_ok=True)
                        continue
                    if already and r.status_code == 200:
                        already = 0
                    if r.status_code not in (200, 206):
                        r.raise_for_status()
                    remaining = int(r.headers.get("content-length") or 0)
                    total = (already + remaining) if r.status_code == 206 else remaining
                    done = already
                    mode = "ab" if r.status_code == 206 else "wb"
                    with open(part, mode) as f:
                        for chunk in r.iter_bytes(1 << 16):
                            f.write(chunk)
                            done += len(chunk)
                            if on_progress and (done % (1 << 20) < (1 << 16)):
                                on_progress(f"正在下载模型（{filename}）…", done, total)
                size = part.stat().st_size if part.exists() else 0
                if total and size < total:
                    if on_progress:
                        on_progress("连接中断，正在断点续传…", size, total)
                    time.sleep(1.0)
                    continue
                if not valid_model(part):
                    raise RuntimeError('下载的文件不是完整 GGUF 模型，请重试')
                if not _matches_checksum(part, item.get("sha256")):
                    part.unlink(missing_ok=True)
                    raise RuntimeError('模型校验失败，正在重新下载')
                part.replace(dest)
                return dest
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5)
                continue
    raise RuntimeError(f"模型下载失败：{last_err}")


def _matches_checksum(path: Path, expected: str | None) -> bool:
    if not expected:
        return True
    try:
        with path.open("rb") as handle:
            return hashlib.file_digest(handle, "sha256").hexdigest() == expected.lower()
    except OSError:
        return False


def engine_dir(variant: str) -> Path:
    return ENGINE_DIR / variant


def server_exe(variant: str) -> Path:
    return engine_dir(variant) / "llama-server.exe"


def installed(variant: str) -> bool:
    if variant not in VARIANTS:
        return False
    directory = engine_dir(variant)
    required = ['llama-server.exe', 'llama-server-impl.dll', 'llama.dll', 'ggml.dll', 'ggml-base.dll']
    if variant == 'vulkan':
        required.append('ggml-vulkan.dll')
    elif variant == 'cuda':
        required += ['ggml-cuda.dll', 'cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll']
    def nonempty(path):
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False
    return all(nonempty(directory / name) for name in required) and any(
        nonempty(path) for path in directory.glob('ggml-cpu*.dll'))


def installed_variants() -> list[str]:
    return [v for v in VARIANTS if installed(v)]


_nvidia_cache: bool | None = None


def preferred_variant(cfg, fallback: str = "cpu") -> str:
    """当前该用哪个引擎：先看设置里选的，没装就在已装的里挑（CUDA > Vulkan > CPU）。"""
    want = str(cfg.get("translate.services.llamacpp.variant", "") or "")
    if want and installed(want):
        return want
    for v in (("cuda", "vulkan", "cpu") if nvidia_available() else ("vulkan", "cpu")):
        if installed(v):
            return v
    return want or fallback


def nvidia_available() -> bool:
    """粗略判断这台机器有没有 NVIDIA 显卡（用来决定要不要显示 CUDA 引擎）。"""
    global _nvidia_cache
    if _nvidia_cache is None:
        import shutil

        ok = bool(shutil.which("nvidia-smi"))
        if not ok:
            sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
            ok = any((sys32 / n).exists() for n in ("nvcuda.dll", "nvml.dll"))
        _nvidia_cache = ok
    return _nvidia_cache


def download_engine(variant: str, on_progress: ProgressCb = None) -> Path:
    """Binary engines are serviced through the offline distribution, never by a mirror."""
    if variant not in VARIANTS:
        raise ValueError(f"未知引擎版本：{variant}")
    if installed(variant):
        return engine_dir(variant)
    raise RuntimeError("本地引擎缺失或不完整，请退出译声后使用完整发行包修复。不会从镜像下载可执行程序。")


# ---------------------------------------------------------------- 模型管理
def list_gguf(folder: Path | None = None) -> list[Path]:
    d = folder or MODELS_LLM
    if not d.exists():
        return []
    return sorted((p for p in d.glob("*.gguf") if valid_model(p)), key=lambda p: p.stat().st_size)


def valid_model(path: Path) -> bool:
    try:
        if path.stat().st_size < 1024 * 1024:
            return False
        with path.open('rb') as f:
            return f.read(4) == b'GGUF'
    except OSError:
        return False


def resolve_model(preferred: str = "") -> Path | None:
    """确定要加载的 GGUF：优先用户指定，其次 models/llm 里体积最小的一个。"""
    if preferred:
        p = Path(preferred)
        if not p.is_absolute():
            p = ROOT / p
        if p.suffix.lower() == ".gguf" and valid_model(p):
            return p
        moved = MODELS_LLM / Path(preferred).name
        if valid_model(moved):
            return moved
    models = list_gguf()
    return models[0] if models else None


def ollama_blobs_dir() -> Path | None:
    d = Path(os.environ.get("USERPROFILE", "")) / ".ollama" / "models" / "blobs"
    return d if d.exists() else None


def list_ollama_gguf() -> list[Path]:
    """列出 Ollama 里已下载、且是标准 GGUF 的模型文件。"""
    out: list[Path] = []
    d = ollama_blobs_dir()
    if not d:
        return out
    for f in d.iterdir():
        try:
            if not f.is_file() or f.stat().st_size < 50 * 1024 * 1024:
                continue
            with open(f, "rb") as fh:
                if fh.read(4) == b"GGUF":
                    out.append(f)
        except Exception:  # noqa: BLE001
            continue
    return sorted(out, key=lambda p: -p.stat().st_size)


def import_gguf(src: Path, name: str = "", on_progress: ProgressCb = None,
                copy: bool = True) -> Path:
    """把外部 GGUF 导入 models/llm（默认复制，避免删源文件后失效）。"""
    src = Path(src)
    if not src.exists():
        raise RuntimeError(f"找不到模型文件：{src}")
    with open(src, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise RuntimeError("这不是标准的 GGUF 模型文件")
    if not copy:
        return src
    MODELS_LLM.mkdir(parents=True, exist_ok=True)
    stem = (name or src.stem)[:48] or "model"
    dst = MODELS_LLM / f"{stem}.gguf"
    if src.resolve() == dst.resolve():
        return src
    total = src.stat().st_size
    done = 0
    with open(src, "rb") as rf, open(dst, "wb") as wf:
        while True:
            buf = rf.read(1 << 22)
            if not buf:
                break
            wf.write(buf)
            done += len(buf)
            if on_progress and (done % (64 << 20) < (4 << 20)):
                on_progress(f"正在复制模型（{dst.name}）…", done, total)
    return dst


# ---------------------------------------------------------------- 服务器
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------- 随父进程退出
_JOB_HANDLE: int | None = None
_JOB_LOCK = threading.Lock()
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


class _IoCounters(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _BasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class _ExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _keep_alive_job() -> int | None:
    """建一个「父进程一关就把子进程一起带走」的作业对象（Windows Job Object）。

    以前软件被强杀 / 崩溃时，llama-server 会留在后台：它占着 3~4GB 内存和显存，
    而新启动的引擎因为分不到显存直接加载失败（engine/server.log 里的
    OutOfDeviceMemory）。挂上这个作业对象后，父进程一消失子进程就会跟着退出。
    """
    global _JOB_HANDLE
    if os.name != "nt":
        return None
    with _JOB_LOCK:
        if _JOB_HANDLE is None:
            try:
                k32 = ctypes.WinDLL("kernel32", use_last_error=True)
                k32.CreateJobObjectW.restype = ctypes.c_void_p
                k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
                handle = k32.CreateJobObjectW(None, None)
                if not handle:
                    return None
                info = _ExtendedLimitInformation()
                info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                if not k32.SetInformationJobObject(
                        ctypes.c_void_p(handle), _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                        ctypes.byref(info), ctypes.sizeof(info)):
                    return None
                _JOB_HANDLE = int(handle)
            except Exception:  # noqa: BLE001
                return None
        return _JOB_HANDLE


def _join_job(pid: int) -> None:
    """把刚启动的引擎进程加进作业对象（失败也不影响正常使用）。"""
    handle = _keep_alive_job()
    if not handle:
        return
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        PROCESS_SET_QUOTA, PROCESS_TERMINATE = 0x0100, 0x0001
        proc = k32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, False, int(pid))
        if not proc:
            return
        try:
            k32.AssignProcessToJobObject(ctypes.c_void_p(handle), ctypes.c_void_p(proc))
        finally:
            k32.CloseHandle(ctypes.c_void_p(proc))
    except Exception:  # noqa: BLE001
        pass


def stale_servers(dry_run: bool = False) -> list[int]:
    """处理上次没退干净、还占着内存和显存的引擎进程，返回被结束的 PID。

    只认「本软件 engine 目录下」的 llama-server.exe，其它程序一概不碰。
    """
    if psutil is None:
        return []
    held = {srv._proc.pid for srv in list(_SERVERS.values()) if srv._proc is not None}
    me = os.getpid()
    try:
        root = str(ENGINE_DIR.resolve()).lower()
    except Exception:  # noqa: BLE001
        root = str(ENGINE_DIR).lower()
    found: list[int] = []
    for proc in psutil.process_iter(["pid", "name", "exe"]):
        try:
            pid = int(proc.info.get("pid") or 0)
            if pid in held or pid == me:
                continue
            # Preview and stable share binaries: never stop another build's process.
            if proc.ppid() != me:
                continue
            exe = str(proc.info.get("exe") or "")
            if not exe or os.path.basename(exe).lower() != "llama-server.exe":
                continue
            if Path(exe).resolve().parent.parent != ENGINE_DIR.resolve():
                continue
            found.append(pid)
            if dry_run:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess,
                ValueError, OSError):
            continue
    return found


def _diagnose_start_failure(err: Exception) -> str:
    """把引擎启动失败翻译成人话（日志里最常见的是显存不足）。"""
    tail = ""
    try:
        log = SERVER_LOG
        if log.exists():
            with log.open("rb") as handle:
                handle.seek(max(0, log.stat().st_size - 8192))
                tail = handle.read(8192).decode("utf-8", "ignore").lower()
    except Exception:  # noqa: BLE001
        tail = ""
    if "outofdevicememory" in tail or "failed to allocate" in tail:
        return ("显卡显存/内存不足，模型没能载入。可到「设置 → 翻译服务 → 用哪个引擎」"
                "换成 CPU 引擎，或关掉占显存的其它程序后重试")
    if "failed to create context" in tail or "out of memory" in tail:
        return "模型载入时内存不够，没能建起上下文（可换小一点的模型或用 CPU 引擎）"
    if "shared object initialization failed" in tail:
        return "CUDA 计算模块初始化失败，请释放其它模型后重试；当前备用引擎可继续使用"
    return f"引擎启动失败：{err}"


def _trim_startup_working_set(pid: int) -> bool:
    """Trim once after GPU upload; mapped CPU weights fault back in as needed."""
    if os.name != "nt":
        return False
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong]
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.K32EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        k32.K32EmptyWorkingSet.restype = ctypes.c_bool
        k32.CloseHandle.argtypes = [ctypes.c_void_p]
        # Only our child process: QUERY_INFORMATION | SET_QUOTA.
        handle = k32.OpenProcess(0x0400 | 0x0100, False, int(pid))
        if not handle:
            return False
        try:
            return bool(k32.K32EmptyWorkingSet(handle))
        finally:
            k32.CloseHandle(handle)
    except (AttributeError, OSError):
        return False


def _runtime_profile(model: Path) -> dict:
    """Optional per-model local tuning; unknown models retain the defaults."""
    import json
    try:
        profiles = json.loads((ENGINE_DIR / "runtime_profiles.json").read_text("utf-8"))
        value = profiles.get(model.name.lower(), {})
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError, AttributeError):
        return {}


def _profile_int(profile: dict, key: str, default: int, lower: int, upper: int) -> int:
    try:
        value = int(profile.get(key, default))
    except (TypeError, ValueError, OverflowError):
        value = default
    return max(lower, min(upper, value))


class LlamaServer:
    """管理一个 llama-server 进程。"""

    def __init__(self, model: Path, variant: str = "cpu", ctx: int = 2048, threads: int = 0, mmproj: Path | None = None):
        self.model = Path(model)
        self.profile = _runtime_profile(self.model)
        self.variant = variant
        self.ctx = _profile_int(self.profile, "context", ctx, 512, 8192)
        self.mmproj = mmproj
        self.threads = _profile_int(self.profile, "threads", threads or max(4, min(10, (os.cpu_count() or 8) // 2)), 1, 16)
        self.port = _free_port()
        self._proc: subprocess.Popen | None = None
        self._health_at = 0.0
        self._health_ok = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def start(self, on_status: Optional[Callable[[str], None]] = None, timeout: float = 240.0,
              checkpoint: Optional[Callable[[], None]] = None) -> None:
        if checkpoint:
            checkpoint()
        exe = server_exe(self.variant)
        if not exe.exists():
            raise RuntimeError(f"引擎缺失或不完整（{self.variant}），请使用完整发行包修复。")
        if not self.model.exists():
            raise RuntimeError(f"模型文件不存在：{self.model}")
        def number(key, default, lower, upper):
            return _profile_int(self.profile, key, default, lower, upper)
        args = [
            str(exe), "-m", str(self.model), "-c", str(self.ctx), "-t", str(self.threads),
            "--host", "127.0.0.1", "--port", str(self.port),
            "-np", "1",
            "-b", str(number("batch", 128 if self.variant in GPU_VARIANTS else 256, 32, 1024)),
            "-ub", str(number("ubatch", 64 if self.variant in GPU_VARIANTS else 128, 32, 512)),
            "--fit-target", str(number("fit_margin", 512 if self.variant == "cuda" else (768 if self.variant in GPU_VARIANTS else 256), 256, 2048)),
        ]
        if self.mmproj is not None:
            args += ["--mmproj", str(self.mmproj)]
        if self.variant in GPU_VARIANTS:
            args += ["-ngl", "auto", "--no-warmup"]
        if self.variant == "cuda":
            # Keep file-backed CPU weights instead of a large pinned allocation.
            args += ["--load-mode", "mmap", "--cache-ram", "256"]
        if self.variant in GPU_VARIANTS and self.profile:
            cache = self.profile.get("kv_cache", "f16")
            if cache not in ("f16", "q8_0"):
                cache = "f16"
            args += ["-fa", "on", "-ctk", cache, "-ctv", cache,
                     "-tb", str(number("threads_batch", 8, 1, 16))]
        if self.profile.get("chat_template") == "chatml":
            args += ["--chat-template", "chatml", "--reasoning-format", "none"]
        child_env = None
        if self.variant == "cuda" and os.name == "nt":
            child_env = os.environ.copy()
            # Avoid the optional PDL probe implicated in Windows CUDA init failures.
            child_env.setdefault("GGML_CUDA_PDL", "0")
        SERVER_LOG.parent.mkdir(parents=True, exist_ok=True)
        log = open(SERVER_LOG, "ab")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if on_status:
            on_status(f"正在启动本地引擎（{self.variant}），加载模型可能需要 5~30 秒…")
        try:
            self._proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                                          creationflags=flags, env=child_env)
        finally:
            log.close()
        _join_job(self._proc.pid)
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if checkpoint:
                checkpoint()
            if self._proc.poll() is not None:
                raise RuntimeError("引擎进程已退出")
            if self.is_up():
                if self.variant == "cuda":
                    _trim_startup_working_set(self._proc.pid)
                return
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"引擎启动超时（详情见 {SERVER_LOG}）")

    def is_up(self) -> bool:
        # An exited child is detected immediately; healthy local children need
        # only an occasional HTTP probe, rather than one before every preview.
        if self._proc is None or self._proc.poll() is not None:
            self._health_ok = False
            return False
        now = time.monotonic()
        if self._health_ok and now - self._health_at < 15.0:
            return True
        try:
            r = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1.5, trust_env=False)
            self._health_ok = r.status_code == 200
        except Exception:  # noqa: BLE001
            self._health_ok = False
        self._health_at = time.monotonic()
        return self._health_ok

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        self._health_ok = False
        self._health_at = 0.0
        if proc is None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=6)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=6)
        except Exception:  # noqa: BLE001
            pass


_SERVERS: dict[str, LlamaServer] = {}
_LOCK = threading.Lock()
_RESET_REQUESTED = threading.Event()
# 启动失败的记录：{model|variant: (失败时间, 说明)}。失败后一段时间内直接报错，
# 不要每翻译一句就重新拉一个引擎进程（那正是会越跑越卡的元凶）。
_FAILED: dict[str, tuple[float, str]] = {}
_FAILED_COOLDOWN = 120.0


def running_variant(model: Path, variants: list[str], *, ctx: int = 2048,
                    checkpoint: Optional[Callable[[], None]] = None) -> str | None:
    """Return a healthy already-owned engine, without starting another model.

    Reusing a fallback prevents periodic cold-load retries while watching a
    stream. stop_translation() clears ownership, so changing settings still
    retries the requested backend normally.
    """
    if checkpoint:
        checkpoint()
    while not _LOCK.acquire(timeout=0.1):
        if checkpoint:
            checkpoint()
    try:
        for variant in variants:
            if checkpoint:
                checkpoint()
            key = f"{model.resolve()}|{variant}|{ctx}|"
            server = _SERVERS.get(key)
            if server is not None and server.is_up():
                if checkpoint:
                    checkpoint()
                return variant
        return None
    finally:
        _LOCK.release()


def ensure_running(model: Path, variant: str = "cpu",
                   on_status: Optional[Callable[[str], None]] = None,
                   *, mmproj: Path | None = None, ctx: int = 2048,
                   checkpoint: Optional[Callable[[], None]] = None) -> str:
    """确保引擎在运行并返回 OpenAI 兼容的 base_url（同一模型只启动一次）。"""
    caller_checkpoint = checkpoint
    def checkpoint():
        if caller_checkpoint:
            caller_checkpoint()
        if _RESET_REQUESTED.is_set():
            from .translate import TranslationCancelled
            raise TranslationCancelled()
    key = f"{model.resolve()}|{variant}|{ctx}|{mmproj.resolve() if mmproj else ''}"
    # A preview may become obsolete while another request loads the model.
    # Check while waiting, rather than blocking cancellation behind that load.
    if checkpoint:
        checkpoint()
    while not _LOCK.acquire(timeout=0.1):
        if checkpoint:
            checkpoint()
    try:
        if checkpoint:
            checkpoint()
        hit = _FAILED.get(key)
        if hit is not None and time.monotonic() - hit[0] < _FAILED_COOLDOWN:
            raise RuntimeError(hit[1])
        srv = _SERVERS.get(key)
        if srv is not None and srv.is_up():
            return srv.base_url
        if srv is not None:
            srv.stop()
        # 先把上次没退干净的引擎收掉，否则显存/内存被老进程占着，新引擎会加载失败
        stale_servers()
        srv = LlamaServer(model, variant, ctx=ctx, mmproj=mmproj)
        try:
            srv.start(on_status=on_status, timeout=60.0 if variant in GPU_VARIANTS else 120.0,
                      checkpoint=checkpoint)
        except Exception as e:  # noqa: BLE001
            srv.stop()
            # Cancellation is not an engine failure and must not poison retries.
            if checkpoint:
                checkpoint()
            msg = _diagnose_start_failure(e)
            _FAILED[key] = (time.monotonic(), msg)
            raise RuntimeError(msg) from e
        _FAILED.pop(key, None)
        _SERVERS[key] = srv
        return srv.base_url
    finally:
        _LOCK.release()


def stop_translation() -> None:
    """切换翻译配置时保留正在使用的语音识别服务。"""
    # Interrupt cold loading before waiting for its lock (formerly up to 120s).
    _RESET_REQUESTED.set()
    try:
        with _LOCK:
            for key, server in list(_SERVERS.items()):
                if server.mmproj is None:
                    server.stop()
                    del _SERVERS[key]
            _FAILED.clear()
    finally:
        _RESET_REQUESTED.clear()


def stop_all() -> None:
    _RESET_REQUESTED.set()
    try:
        with _LOCK:
            for srv in list(_SERVERS.values()):
                try:
                    srv.stop()
                except Exception:  # noqa: BLE001
                    pass
            _SERVERS.clear()
    finally:
        _RESET_REQUESTED.clear()


atexit.register(stop_all)
