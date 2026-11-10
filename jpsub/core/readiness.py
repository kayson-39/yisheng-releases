"""Cheap offline checks. These detect missing/truncated assets, not model quality."""
from __future__ import annotations

import stat
from pathlib import Path


def file_at_least(path: Path, minimum: int = 1) -> bool:
    try:
        info = path.stat()
        return stat.S_ISREG(info.st_mode) and info.st_size >= minimum
    except OSError:
        return False


def parakeet_ready(directory: Path) -> bool:
    # The bundled acoustic model is hundreds of MB; don't hash it at UI startup.
    return (file_at_least(directory / "model.int8.onnx", 1024 * 1024)
            and file_at_least(directory / "tokens.txt"))


def vad_ready(path: Path) -> bool:
    return file_at_least(path, 100_000)


def live_provider_for(cfg) -> str:
    """Select the preview provider without contacting a model or service."""
    mode = str(cfg.get("translate.live_provider", "auto"))
    main = str(cfg.get("translate.provider", "ms-free"))
    if mode == "same":
        return main
    if mode == "auto":
        from .config import TRANSLATE_PRESETS

        kind = TRANSLATE_PRESETS.get(main, {}).get("kind")
        model = str(cfg.service(main).get('model') or '').lower()
        if kind == 'llamacpp' and any(name in model for name in ('hy-mt2', 'galtransl')):
            return main
        if kind in ("ollama", "llamacpp", "localmt"):
            return "localmt"
        return main
    return mode


MODEL_NEXT_STEP = "设置 → 翻译：下载 Sakura 大模型（约 4.25 GB）、选择已有 GGUF，或改用本地轻量翻译。不会自动下载。"


def speech_checks() -> list[tuple[str, bool]]:
    from ..paths import MODELS
    from .asr import FinalASR

    return [("日语高速识别 Parakeet", FinalASR.available()),
            ("静音检测 silero_vad.onnx", vad_ready(MODELS / "silero_vad.onnx"))]


def lightweight_ready() -> bool:
    from .local_mt import installed
    return installed("ja", "zh") or (installed("ja", "en") and installed("en", "zh"))


def installation_checks(cfg) -> list[tuple[str, bool | None]]:
    """None means optional and absent, not a missing/corrupt required component."""
    from . import llama_engine as le
    rows: list[tuple[str, bool | None]] = list(speech_checks())
    rows += [("本地翻译引擎文件", bool(le.installed_variants())),
             ("本地轻量翻译模型", lightweight_ready())]
    default = le.MODELS_LLM / le.RECOMMENDED_MODELS[0]["name"]
    if default.exists() and not le.valid_model(default):
        rows.append(("推荐 GGUF 文件损坏，请重新下载：" + default.name, False))
    elif le.resolve_model(cfg.service("llamacpp").get("model") or "") is not None:
        rows.append(("本地翻译 GGUF 模型", True))
    else:
        rows.append(("可选 GGUF 模型未安装。" + MODEL_NEXT_STEP, None))
    return rows


def local_checks(cfg) -> list[tuple[str, bool]]:
    """Check only the local assets required by the current listening configuration."""
    rows = speech_checks()
    if not cfg.get("translate.enabled", True):
        return rows
    providers = {cfg.get("translate.provider", "llamacpp")}
    if (cfg.get("translate.live", True) and cfg.get("asr.partial", True)
            and cfg.get("asr.draft_mode", "rolling") != "off"):
        providers.add(live_provider_for(cfg))
    if "llamacpp" in providers:
        from . import llama_engine
        rows.extend([
            ("本地翻译 GGUF 模型", llama_engine.resolve_model(cfg.service("llamacpp").get("model") or "") is not None),
            ("本地翻译引擎文件", bool(llama_engine.installed_variants())),
        ])
    if "localmt" in providers:
        rows.append(("本地轻量翻译模型", lightweight_ready()))
    return rows
