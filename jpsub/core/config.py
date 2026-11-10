# -*- coding: utf-8 -*-
"""配置管理：默认值、JSON 持久化、翻译/识别服务预设。"""
from __future__ import annotations

import copy
import json
import threading
import os
import tempfile
import shutil
import math
from pathlib import Path
from typing import Any

from ..paths import CONFIG_PATH, ROOT, USER_DATA

# ---------------------------------------------------------------- 服务预设
# kind: openai=OpenAI 兼容 chat/completions；deepl；googlefree；msfree；mymemory
TRANSLATE_PRESETS: dict[str, dict] = {
    "ms-free": dict(
        label="免费 · 微软翻译（免密钥）", kind="msfree", base_url="", model="",
        key_hint="无需密钥，直连可用", unofficial=True,
    ),
    "google-free": dict(
        label="免费 · 谷歌翻译（免密钥）", kind="googlefree", base_url="", model="",
        key_hint="无需密钥，需能访问谷歌", unofficial=True,
    ),
    "mymemory-free": dict(
        label="免费 · MyMemory（免密钥）", kind="mymemory", base_url="", model="",
        key_hint="无需密钥，有每日额度限制", unofficial=True,
    ),
    "localmt": dict(
        label="本地 · 轻量翻译（低配备用）", kind="localmt", base_url="", model="",
        key_hint="完全离线，日语经英语中转为中文；速度快，口语和反问句精度有限",
    ),
    "llamacpp": dict(
        label="本地 · 自然口语翻译", kind="llamacpp",
        base_url="", model="",
        key_hint="自带引擎，不依赖 Ollama；选好 GGUF 模型即可用，面板里可一键导入 Ollama 已下载的模型",
    ),
    "ollama": dict(
        label="本地 · Ollama（可自定义任意模型）", kind="ollama",
        base_url="http://127.0.0.1:11434", model="translategemma:4b",
        key_hint="需要安装 Ollama；可自由选择/下载本地模型，面板里带测速",
    ),
    "deepseek": dict(
        label="DeepSeek（推荐，便宜且中文好）", kind="openai",
        base_url="https://api.deepseek.com/v1", model="deepseek-chat",
        key_hint="在 platform.deepseek.com 创建 API Key",
    ),
    "siliconflow": dict(
        label="硅基流动 SiliconFlow", kind="openai",
        base_url="https://api.siliconflow.cn/v1", model="deepseek-ai/DeepSeek-V3",
        key_hint="在 siliconflow.cn 创建 API Key（有免费额度）",
    ),
    "dashscope": dict(
        label="阿里云百炼 · 通义千问", kind="openai",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", model="qwen-flash",
        key_hint="在 bailian.console.aliyun.com 创建 API Key",
    ),
    "zhipu": dict(
        label="智谱 GLM", kind="openai",
        base_url="https://open.bigmodel.cn/api/paas/v4", model="glm-4-flash",
        key_hint="在 bigmodel.cn 创建 API Key",
    ),
    "moonshot": dict(
        label="月之暗面 Kimi", kind="openai",
        base_url="https://api.moonshot.cn/v1", model="moonshot-v1-8k",
        key_hint="在 platform.moonshot.cn 创建 API Key",
    ),
    "openai": dict(
        label="OpenAI / 兼容代理", kind="openai",
        base_url="https://api.openai.com/v1", model="gpt-4o-mini",
        key_hint="需要能访问 api.openai.com，或填代理地址",
    ),
    "gemini": dict(
        label="Google Gemini", kind="openai",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model="gemini-2.5-flash", key_hint="在 aistudio.google.com 获取 API Key",
    ),
    "deepl": dict(
        label="DeepL", kind="deepl", base_url="", model="",
        key_hint="DeepL 免费版 Key 即可，直连质量高",
    ),
    "custom": dict(
        label="自定义（本地 LM Studio / llama.cpp / 云端均可）", kind="openai",
        base_url="", model="", key_hint="填写任意 OpenAI 兼容的 Base URL / 模型 / Key；本地服务可点“拉取模型列表”",
    ),
}

# 日语识别只保留一个经过取舍的本地模型，避免用户在相似选项中反复试错。
FINAL_ENGINES: dict[str, dict] = {
    "parakeet": dict(label="日语高速识别（Parakeet 0.6B）",
                     path="parakeet-ja-0.6b-int8", kind="sherpa", download="parakeet-ja"),
}


def _defaults() -> dict:
    return {
        "version": 1,
        "model_setup_version": 5,
        "audio": {"source": "loopback", "device": "", "block_ms": 80},
        "vad": {"threshold": 0.4, "min_silence_ms": 350, "min_speech_ms": 200, "max_segment_s": 10},
        "asr": {
            "partial": True,
            "draft_mode": "rolling",     # rolling=滚动识别 / off=关闭实时草稿
            "draft_interval_ms": 500,
            "draft_threads": 4,
            "final_engine": "parakeet",
            "final_threads": 6,
            "max_segment_s": 10,
            "context_prompt": False,
        },
        "translate": {
            "enabled": True,
            "provider": "llamacpp",
            "auto_fallback": False,
            "live_provider": "same",
            "live": True,
            "live_min_interval_ms": 450,
            "live_min_new_chars": 2,
            "context_lines": 3,
            "glossary": [],
            "services": {"llamacpp": {"model": "models/llm/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf", "variant": "auto"}},
        },
        "text_translate": {"provider": "same", "services": {}},
        "ui": {
            "theme": "light",
            "accent": "#d95482",
            "bg": "none",
            "bg_blur": 16,
            "bg_dim": 45,
            "bg_opacity": 100,
            "font": "Microsoft YaHei UI",
            "jp_size": 13,
            "zh_size": 19,
            "max_lines": 100,
            "display": "both",  # both / zh / jp
            "show_jp_in_history": True,
            "window": {"w": 1220, "h": 720, "x": None, "y": None, "on_top": False},
        },
        "overlay": {
            "enabled": False, "lines": 0, "zh_size": 24, "jp_size": 15,
            "theme": "dark", "opacity": 88, "text_shadow": True, "show_jp": True,
            "click_through": False, "locked": False,
            "x": None, "y": None, "w": 760, "h": 360,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _validated_user(user: dict) -> dict:
    """Repair malformed known fields without discarding valid custom settings."""
    if not isinstance(user, dict):
        raise ValueError("配置根节点必须是对象")

    def validate(values, defaults):
        result = copy.deepcopy(values)
        for key, default in defaults.items():
            if key not in result:
                continue
            value = result[key]
            if isinstance(default, dict):
                result[key] = validate(value, default) if isinstance(value, dict) else {}
            elif isinstance(default, bool):
                if isinstance(value, str) and value.lower() in ("true", "false"):
                    result[key] = value.lower() == "true"
                elif not isinstance(value, bool):
                    result[key] = default
            elif isinstance(default, (int, float)):
                try:
                    number = float(value)
                    if isinstance(value, bool) or not math.isfinite(number):
                        raise ValueError("invalid number")
                    result[key] = int(number) if isinstance(default, int) else number
                except (ValueError, TypeError, OverflowError):
                    result[key] = default
            elif isinstance(default, str) and not isinstance(value, str):
                result[key] = default
            elif isinstance(default, list) and not isinstance(value, list):
                result[key] = []
            elif default is None and value is not None:
                # Optional window coordinates are the only null defaults.
                try:
                    result[key] = int(value)
                except (ValueError, TypeError, OverflowError):
                    result[key] = None
        return result

    user = validate(user, _defaults())
    for section in ("translate", "text_translate"):
        values = user.get(section, {})
        services = values.get("services", {})
        for provider, service in list(services.items()):
            services[provider] = validate(service, {
                "model": "", "base_url": "", "api_key": "", "variant": "auto",
            }) if isinstance(service, dict) else {}
    translation = user.get("translate", {})
    if "glossary" in translation:
        from .glossary import normalize
        rows = translation["glossary"]
        usable = [row for row in rows if isinstance(row, (list, tuple)) and len(row) == 2
                  and all(isinstance(value, str) for value in row)]
        translation["glossary"] = normalize(usable)
    return user


class Config:
    """线程安全的配置对象（dict 风格访问）。"""

    def __init__(self, path: Path = CONFIG_PATH, *, migrate_legacy: bool = True):
        self.path = Path(path)
        self._lock = threading.RLock()
        self.data = _defaults()
        self.load(migrate_legacy=migrate_legacy)

    # ------------------------------------------------------------------
    def load(self, *, migrate_legacy: bool = True) -> None:
        backup = self.path.with_suffix('.json.bak')
        if migrate_legacy and self.path == CONFIG_PATH and not self.path.exists() and not backup.exists():
            self._migrate_legacy()
        for candidate in (self.path, backup):
            try:
                user = _validated_user(json.loads(candidate.read_text(encoding="utf-8")))
                if int(user.get('model_setup_version', 0)) < 3:
                    # 旧版本的高门限和备用识别设置会让轻声、短句无法进入识别。
                    user.setdefault('vad', {}).update(threshold=0.4, min_silence_ms=500, max_segment_s=18)
                    asr = user.setdefault('asr', {})
                    asr.update(final_engine='parakeet', max_segment_s=18, context_prompt=False)
                    for name in ('cloud', 'streaming', 'stream_model'):
                        asr.pop(name, None)
                    translation = user.setdefault('translate', {})
                    translation['live_provider'] = 'same'
                    local = translation.setdefault('services', {}).setdefault('llamacpp', {})
                    if not local.get('model') or 'translategemma' in str(local['model']).lower():
                        local['model'] = 'models/llm/Hy-MT2-1.8B-Q4_K_M.gguf'
                    user['model_setup_version'] = 3
                if int(user.get('model_setup_version', 0)) < 4:
                    user.setdefault('asr', {}).update(final_engine='parakeet', draft_interval_ms=500, max_segment_s=10)
                    user.setdefault('vad', {}).update(min_silence_ms=350, max_segment_s=10)
                    user.setdefault('translate', {}).update(live_provider='same', live_min_interval_ms=450, live_min_new_chars=2)
                    local = user.setdefault('translate', {}).setdefault('services', {}).setdefault('llamacpp', {})
                    if not local.get('model') or 'hy-mt2-1.8b' in str(local['model']).lower():
                        local['model'] = 'models/llm/Hy-MT2-7B-Q4_K_M.gguf'
                    user['model_setup_version'] = 4
                if int(user.get('model_setup_version', 0)) < 5:
                    translation = user.setdefault('translate', {})
                    local = translation.setdefault('services', {}).setdefault('llamacpp', {})
                    previous = str(local.get('model') or '').replace('\\', '/').rsplit('/', 1)[-1].lower()
                    bundled = {'hy-mt2-7b-q4_k_m.gguf', 'hy-mt2-1.8b-q4_k_m.gguf',
                               'galtransl-v4-4b-2601.gguf', 'sakura-galtransl-7b-v3.7-iq4_xs.gguf'}
                    if translation.get('provider', 'llamacpp') == 'llamacpp' and (not previous or previous in bundled):
                        local['model'] = 'models/llm/Sakura-Galtransl-7B-v3.7-IQ4_XS.gguf'
                        local.setdefault('variant', 'auto')
                    user['model_setup_version'] = 5
                with self._lock:
                    self.data = _deep_merge(_defaults(), user)
                return
            except Exception:
                continue

    def _migrate_legacy(self):
        """Only known app config files; never scan unrelated user files."""
        from .glossary import merge_glossary, normalize
        local = Path(os.environ.get('LOCALAPPDATA', str(Path.home())))
        candidates = [ROOT/'config.json', local/'Programs/YishengTranslatorWhite/app/config.json',
                      local/'Programs/YishengTranslator/app/config.json']
        valid=[]
        for path in dict.fromkeys(candidates):
            if path.is_file() and path.resolve()!=self.path.resolve():
                try:
                    data=json.loads(path.read_text('utf-8'))
                    if isinstance(data,dict):valid.append((path.stat().st_mtime,path,data))
                except (OSError, ValueError):continue
        if not valid:return
        valid.sort(key=lambda item:item[0])
        data=copy.deepcopy(valid[-1][2]);terms=[]
        backup_dir=USER_DATA/'migration-backups';backup_dir.mkdir(parents=True,exist_ok=True)
        for i,(_,path,old) in enumerate(valid):
            shutil.copy2(path,backup_dir/f'legacy-{i}-config.json')
            translation=old.get('translate')
            rows=translation.get('glossary',[]) if isinstance(translation,dict) else []
            # Keep usable rows from older hand-edited files; preserve the raw
            # source above so malformed rows can still be recovered manually.
            usable=[]
            for row in rows if isinstance(rows,list) else []:
                try:usable.extend(normalize([row]))
                except (ValueError,TypeError):continue
            terms,_=merge_glossary(terms,usable,True)
        if not isinstance(data.get('translate'),dict):data['translate']={}
        data.setdefault('translate',{})['glossary']=terms
        data['data_migration_version']=1
        self.data=_deep_merge(_defaults(),data)
        self.save()

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            text = json.dumps(self.data, ensure_ascii=False, indent=2)
            tmp=None
            try:
                with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=self.path.parent,prefix='.config-',suffix='.tmp',delete=False) as f:
                    tmp=Path(f.name);f.write(text);f.flush();os.fsync(f.fileno())
                if self.path.exists():
                    try:
                        if not isinstance(json.loads(self.path.read_text('utf-8')), dict):
                            raise ValueError('配置根节点必须是对象')
                        shutil.copy2(self.path,self.path.with_suffix('.json.bak'))
                    except (OSError,ValueError):pass
                os.replace(tmp,self.path)
            finally:
                if tmp and tmp.exists():tmp.unlink()

    # ------------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self.data
        for part in path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def set(self, path: str, value: Any, save: bool = True) -> None:
        with self._lock:
            node = self.data
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        if save:
            self.save()

    def update_many(self, mapping: dict[str, Any], save: bool = True) -> None:
        for k, v in mapping.items():
            self.set(k, v, save=False)
        if save:
            self.save()

    # ------------------------------------------------------------------
    def service(self, provider: str) -> dict:
        """返回某个翻译服务的有效配置（预设 + 用户覆盖）。"""
        preset = TRANSLATE_PRESETS.get(provider, TRANSLATE_PRESETS["custom"])
        user = self.get(f"translate.services.{provider}", {}) or {}
        merged = {
            "provider": provider,
            "kind": preset["kind"],
            "label": preset["label"],
            "base_url": user.get("base_url") or preset.get("base_url", ""),
            "model": user.get("model") or preset.get("model", ""),
            "api_key": user.get("api_key", ""),
        }
        return merged

    def set_service(self, provider: str, **fields: Any) -> None:
        cur = dict(self.get(f"translate.services.{provider}", {}) or {})
        cur.update(fields)
        self.set(f"translate.services.{provider}", cur)
