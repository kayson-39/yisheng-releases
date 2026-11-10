# -*- coding: utf-8 -*-
"""翻译服务：OpenAI 兼容大模型（流式）、DeepL、以及免密钥便民通道。"""
from __future__ import annotations

import json
import re
import time
from typing import Callable, Iterable, Optional

import httpx

from .http_util import make_client

OnDelta = Optional[Callable[[str], None]]

class TranslationCancelled(Exception):
    """An obsolete preview was cancelled; never trigger provider fallback."""


class TranslationDeadlineExceeded(TimeoutError):
    pass


class TranslationOutputError(RuntimeError):
    """Malformed local output; retry the same model without prior history."""


class TranslationStream:
    """Cooperative cancellation and an overall deadline, independent of SSE activity."""
    def __init__(self, callback, cancelled, seconds=45.0, read_timeout=15.0):
        self.callback = callback
        self.cancelled = cancelled
        self.deadline = time.monotonic() + seconds
        self.read_timeout = read_timeout

    def checkpoint(self):
        if self.cancelled():
            raise TranslationCancelled()
        if time.monotonic() >= self.deadline:
            raise TranslationDeadlineExceeded("翻译请求超过等待上限")

    def __call__(self, piece):
        self.checkpoint()
        self.callback(piece)


def _checkpoint(callback):
    if hasattr(callback, "checkpoint"):
        callback.checkpoint()


def _stream_timeout(callback, default):
    _checkpoint(callback)
    # Loading the model / waiting for its lock may have spent most of the
    # request budget already. Do not start another full-length blocking read.
    deadline = getattr(callback, "deadline", None)
    remaining = max(0.001, deadline - time.monotonic()) if deadline is not None else default
    budget = min(default, remaining)
    read = min(budget, float(getattr(callback, "read_timeout", default)))
    return httpx.Timeout(budget, connect=min(8.0, budget), read=read)


_MAX_EVENT_LINE_BYTES = 1024 * 1024
_LINE_END = re.compile(rb'\r\n|[\r\n]')


def _event_lines(response, callback):
    """Read UTF-8 SSE/NDJSON without buffering an unlimited, unfinished line."""
    _checkpoint(callback)
    if response.status_code >= 400:
        # The status is sufficient. An unbounded error page can trickle forever
        # too; don't make a failed request depend on its response body ending.
        response.raise_for_status()
    pending = bytearray()
    ended_cr = False
    for chunk in response.iter_bytes():
        _checkpoint(callback)
        if not chunk:
            continue
        pos = 1 if ended_cr and chunk.startswith(b'\n') else 0
        ended_cr = chunk.endswith(b'\r')
        view = memoryview(chunk)
        for match in _LINE_END.finditer(chunk, pos):
            if len(pending) + match.start() - pos > _MAX_EVENT_LINE_BYTES:
                raise TranslationOutputError('翻译服务返回的单行数据过长')
            pending.extend(view[pos:match.start()])
            line = pending.decode('utf-8', errors='replace')
            pending.clear()
            yield line
            pos = match.end()
        if len(pending) + len(chunk) - pos > _MAX_EVENT_LINE_BYTES:
            raise TranslationOutputError('翻译服务返回的单行数据过长')
        pending.extend(view[pos:])
    _checkpoint(callback)
    if pending:
        yield pending.decode('utf-8', errors='replace')


SYSTEM_PROMPT = (
    "你是专业的日语→简体中文同声传译字幕员，服务对象是正在观看日语直播的中国观众。\n"
    "翻译要求：\n"
    "1. 只输出简体中文译文本身，不要任何解释、注音、原文、引号或说话人标记。\n"
    "2. 口语化、简洁、贴合直播语气；保留情绪与语气词，不要过度书面化。\n"
    "3. 人名、游戏名、角色名、专有名词使用常见中文译名；没有通用译名时保留日文原文。\n"
    "4. 遇到日式网络用语按中文直播圈习惯翻译；数字、时间、单位保持原意。\n"
    "5. 句尾语气（如「～だよ」「～だね」）用自然的中文语气表达，不要机械直译。\n"
    "6. 直播口语里的「～ない？」「～じゃない？」多为反问或撒娇，用「是不是…？」「…吧？」等自然语气；\n"
    "   只有明确否定陈述时才译成否定句。\n"
    "7. 「やばい」「エグい」等词按语境选褒义（太顶了/绝了/不得了）或贬义（糟糕/可怕），不要一律直译。"
)


def build_messages(text: str, context: list[str], glossary: Iterable[tuple[str, str]],
                   source_lang: str = "ja", target_lang: str = "zh") -> list[dict]:
    sys = SYSTEM_PROMPT if (source_lang, target_lang) == ("ja", "zh") else (
        "你是专业的简体中文→日语翻译。只输出自然、准确的日语译文，不要解释、注音、原文或引号。"
        "保留原文的语气、时态、否定、疑问、人称、数字和专有名词，不添加原文没有的内容。"
    )
    terms = [f"{a}→{b}" for a, b in glossary if a and b]
    if terms:
        sys += "\n术语表（必须遵守）：\n" + "\n".join(terms)
    user_parts = []
    ctx = [c for c in context if c]
    if ctx:
        user_parts.append("【上文，仅供理解指代与语气，不要翻译】\n" + "\n".join(ctx[-4:]))
    user_parts.append(("【本次需要翻译的日语】\n" if source_lang == "ja" else "【本次需要翻译的中文】\n") + text)
    user_parts.append("直接输出中文译文：" if target_lang == "zh" else "直接输出日语译文：")
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _friendly_http_error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        detail = ""
        try:
            body = e.response.json()
            detail = str(body.get("error", {}).get("message") or body.get("message") or "")[:120]
        except httpx.ResponseNotRead:
            pass  # Streaming errors use the known status without reading a possibly stalled body.
        except Exception:  # noqa: BLE001
            detail = (e.response.text or "")[:120]
        hints = {
            401: "API Key 无效或未填写",
            403: "没有权限或地区受限",
            404: "接口地址或模型名不对",
            429: "请求太频繁或额度用尽",
            500: "服务端出错", 502: "网关错误", 503: "服务暂时不可用",
        }
        return f"{hints.get(code, f'HTTP {code}')}" + (f"（{detail}）" if detail else "")
    if isinstance(e, httpx.TimeoutException):
        return "请求超时（网络不通或被墙）"
    if isinstance(e, httpx.ConnectError):
        url = ""
        try:
            url = str(e.request.url)
        except Exception:  # noqa: BLE001
            pass
        if "127.0.0.1" in url or "localhost" in url:
            return "本地服务未启动（请先启动 Ollama 或本地模型服务）"
        return "无法连接服务器（检查网络/代理）"
    return str(e)[:160]
class LanguageProvider:
    """每个服务实例固定语言对，避免并发语音和文字请求互相改变方向。"""

    def __init__(self, *, source_lang: str = "ja", target_lang: str = "zh"):
        if (source_lang, target_lang) not in (("ja", "zh"), ("zh", "ja")):
            raise ValueError("仅支持日文与中文之间的双向翻译")
        self.source_lang = source_lang
        self.target_lang = target_lang


# ---------------------------------------------------------------------------
class OllamaNative(LanguageProvider):
    """本地 Ollama（原生 API）：支持 qwen3 关闭思考、模型常驻、流式输出。"""

    kind = "ollama"

    def __init__(self, base_url: str = "", model: str = "", timeout: float = 180.0,
                 *, source_lang: str = "ja", target_lang: str = "zh"):
        super().__init__(source_lang=source_lang, target_lang=target_lang)
        from .ollama import normalize_base

        self.base = normalize_base(base_url)
        self.model = model or ""
        self.timeout = timeout
        if not self.model:
            raise RuntimeError("未指定本地模型（设置 → 翻译服务 → 本地·Ollama → 选择模型）")

    def _messages(self, text: str, context: list[str], glossary) -> list[dict]:
        terms = [f"{a}={b}" for a, b in glossary if a and b]
        if "translategemma" in self.model.lower():
            if self.source_lang == "zh":
                prompt = ("You are a professional Simplified Chinese (zh-Hans) to Japanese (ja) translator. "
                          "Produce only the natural Japanese translation, preserving the original meaning and tone. "
                          "No commentary, explanations or source text.")
                if terms:
                    prompt += "\nGlossary (must follow): " + "; ".join(terms)
                return [{"role": "user", "content": prompt + "\n\n" + text}]
            prompt = (
                "You are a professional Japanese (ja) to Simplified Chinese (zh-Hans) translator. "
                "Accurately convey the meaning and nuances of the original Japanese text while adhering to "
                "Chinese grammar and cultural conventions, in a natural subtitle style. "
                "Produce only the Chinese translation, no additional commentary, no explanations, no source text."
            )
            prompt += (
                " Note: in Japanese live-streaming speech, \"〜ない？/〜じゃない？\" is usually a rhetorical "
                "question (translate as \"是不是…？\"), and \"やばい\" can be positive slang "
                "(\"太顶了/绝了\") depending on context."
            )
            if terms:
                prompt += "\nGlossary (must follow): " + "; ".join(terms)
            return [{"role": "user", "content": prompt + "\n\n" + text}]
        return build_messages(text, context, glossary, self.source_lang, self.target_lang)

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:
        payload = {
            "model": self.model,
            "messages": self._messages(text, context, glossary),
            "stream": True,
            "keep_alive": "30m",
            "options": {"temperature": 0.2, "num_predict": 800},
        }
        if "qwen3" in self.model.lower():
            payload["think"] = False  # qwen3 默认带思考链，拖慢数倍
        parts: list[str] = []
        import json as _json

        _checkpoint(on_delta)
        completed = False
        with make_client(self.base, timeout=_stream_timeout(on_delta, self.timeout)) as client:
            with client.stream("POST", f"{self.base}/api/chat", json=payload) as r:
                for line in _event_lines(r, on_delta):
                    _checkpoint(on_delta)
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except (ValueError, TypeError) as exc:
                        raise TranslationOutputError("本地翻译数据损坏，译文不完整") from exc
                    if not isinstance(obj, dict):
                        raise TranslationOutputError("本地翻译数据格式错误，译文不完整")
                    if obj.get("error"):
                        raise RuntimeError(str(obj["error"]))
                    piece = str((obj.get("message") or {}).get("content") or "")
                    if piece:
                        parts.append(piece)
                        if on_delta:
                            on_delta(piece)
                    if obj.get("done"):
                        if obj.get("done_reason") == "length":
                            raise TranslationOutputError("本地翻译达到输出上限，译文不完整")
                        completed = True
                        break
        _checkpoint(on_delta)
        if not completed:
            raise TranslationOutputError("本地翻译连接提前结束，译文不完整")
        out = "".join(parts).strip().strip('"').strip()
        if not out:
            raise RuntimeError("本地模型没有返回内容")
        return out


class LocalMTProvider(LanguageProvider):
    """内置极速本地翻译（Opus-MT / Argos，无需任何外部软件）。"""

    kind = "localmt"

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:  # noqa: ARG002
        from .local_mt import shared
        if not _localmt_ready(self.source_lang, self.target_lang):
            raise RuntimeError("所选方向的内置翻译模型尚未安装，请选择已可用的本地大模型或在线服务")
        return shared(self.source_lang, self.target_lang).translate(text, on_delta)


class LlamaCppProvider(LanguageProvider):
    """内置 llama.cpp 引擎（llama-server）：本地、免安装、加载 GGUF 模型。"""

    kind = "llamacpp"

    def __init__(self, service: dict):
        self.service = service or {}
        super().__init__(source_lang=self.service.get("source_lang", "ja"),
                         target_lang=self.service.get("target_lang", "zh"))

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:
        from .llama_engine import ensure_running, resolve_model

        model = resolve_model(str(self.service.get("model") or ""))
        if model is None:
            raise RuntimeError("尚未选择 GGUF 模型（设置 → 翻译服务 → 本地·内置引擎）")
        sakura = "galtransl" in model.name.lower()
        if sakura and (self.source_lang, self.target_lang) != ("ja", "zh"):
            raise RuntimeError("Sakura 专用模型仅支持日语译中文；中文译日语请切回混元模型。")
        variant = str(self.service.get("variant") or "cpu")
        from . import llama_engine as _le

        requested = variant
        candidates = [variant]
        if variant != "cpu":
            candidates += [v for v in ("vulkan", "cuda", "cpu") if v != variant and _le.installed(v)]
        # Keep a healthy fallback resident. Retrying the preferred GPU every
        # failure-cooldown period can load a second model beside the fallback,
        # repeatedly stalling subtitles and exhausting VRAM. A manual engine
        # reset (or a dead fallback) restores the original candidate order.
        active = _le.running_variant(model, candidates,
                                     checkpoint=getattr(on_delta, "checkpoint", None))
        if active is not None:
            candidates = [active] + [v for v in candidates if v != active]
        failures = []
        for candidate in candidates:
            if not _le.installed(candidate):
                continue
            _checkpoint(on_delta)
            try:
                base = ensure_running(model, candidate,
                                      checkpoint=getattr(on_delta, "checkpoint", None))
                self.actual_variant = candidate
                self.engine_reason = ("；".join(failures)
                                      or getattr(self, "engine_reason", "")
                                      or f"沿用已加载的 {candidate.upper()} 备用引擎") if candidate != requested else ""
                break
            except (TranslationCancelled, TranslationDeadlineExceeded):
                raise
            except Exception as exc:
                failures.append(f"{candidate.upper()}：{exc}")
        else:
            raise RuntimeError("；".join(failures) or "没有可用的本地翻译引擎")
        if "translategemma" in model.name.lower():
            return TranslateGemma(base, source_lang=self.source_lang, target_lang=self.target_lang).translate(text, context, glossary, on_delta)
        if "hy-mt2" in model.name.lower():
            return HyMT2(base, source_lang=self.source_lang, target_lang=self.target_lang).translate(text, context, glossary, on_delta)
        if sakura:
            v37 = 'sakura-galtransl-7b-v3.7' in model.name.lower()
            return SakuraGalTransl(base, temperature=0.3 if v37 else 0.0,
                                  repeat_penalty=1.1 if v37 else 1.0,
                                  seed=1234 if v37 else None).translate(text, context, glossary, on_delta)
        return OpenAICompat(base, "", model.stem, source_lang=self.source_lang, target_lang=self.target_lang).translate(text, context, glossary, on_delta)


# ---------------------------------------------------------------------------
class TranslateGemma(LanguageProvider):
    """通过 GGUF 内置模板调用 TranslateGemma 的官方翻译协议。"""
    def __init__(self, base: str, *, source_lang: str = "ja", target_lang: str = "zh"):
        super().__init__(source_lang=source_lang, target_lang=target_lang)
        self.base = base.removesuffix('/v1')

    def messages(self, text, context, glossary):
        source = 'zh-Hans' if self.source_lang == 'zh' else 'ja'
        target = 'zh-Hans' if self.target_lang == 'zh' else 'ja'
        return [{'role': 'user', 'content': source + '[sprt]' + target + '[sprt]' + text}]

    def sampling(self):
        return {'temperature': 0.0}

    def check_output(self, source, output, finish=None):
        if finish in ('length', 'empty', 'content_filter'):
            raise TranslationOutputError('本地翻译未正常完成')

    def translate(self, text, context, glossary, on_delta=None):
        # 模型 GGUF 自带的 chat_template 只在收到这一格式时才会生成训练时使用的
        # 对应语言方向的专用提示。旧代码手工拼模板，还额外重复了 BOS，
        # llama.cpp 会明确报警并降低短句翻译的稳定性。
        _ = context, glossary
        payload = {
            'messages': self.messages(text, context, glossary),
            'max_tokens': 512,
            'stream': True,
            **self.sampling(),
        }
        parts = []
        completed = False
        _checkpoint(on_delta)
        with make_client(self.base, timeout=_stream_timeout(on_delta, 90)) as client:
            with client.stream('POST', self.base + '/v1/chat/completions', json=payload) as response:
                for line in _event_lines(response, on_delta):
                    _checkpoint(on_delta)
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        completed = True
                        break
                    try:
                        obj = json.loads(data)
                    except (ValueError, TypeError) as exc:
                        raise TranslationOutputError('本地翻译数据损坏，译文不完整') from exc
                    if not isinstance(obj, dict):
                        raise TranslationOutputError('本地翻译数据格式错误，译文不完整')
                    if obj.get('error'):
                        raise RuntimeError(str(obj['error']))
                    choices = obj.get('choices') or []
                    piece = str((choices[0].get('delta') or {}).get('content') or '') if choices else ''
                    if piece:
                        parts.append(piece)
                        self.check_output(text, ''.join(parts))
                        if on_delta:
                            on_delta(piece)
                    if choices and choices[0].get('finish_reason'):
                        self.check_output(text, ''.join(parts), choices[0]['finish_reason'])
                        completed = True
                        break  # A terminal event is sufficient; do not wait for delayed [DONE].
        _checkpoint(on_delta)
        if not completed:
            raise TranslationOutputError('本地翻译连接提前结束，译文不完整')
        out = ''.join(parts).strip()
        if not out:
            self.check_output(text, out, 'empty')
            raise RuntimeError('本地翻译模型未返回译文')
        return out


class HyMT2(TranslateGemma):
    """腾讯翻译专用模型：用内置聊天模板，支持术语、背景和口语风格。"""

    def messages(self, text, context, glossary):
        sections = []
        terms = [f'{a} 翻译成 {b}' for a, b in glossary if a and b]
        if terms:
            sections.append('参考下面的翻译：\n' + '\n'.join(terms[:40]))
        background = [str(c).strip() for c in context if c]
        if background:
            sections.append('【背景信息，仅供理解，不要翻译】\n' + '\n'.join(background[-2:])[-240:])
        instruction = ('将下面日语翻译成自然、简洁的现代汉语口语字幕。' if self.target_lang == 'zh'
                       else '将下面中文翻译成自然、简洁的现代日语。')
        sections.append(
            instruction + '只翻译已经说出的内容，'
            '原句没说完时保留未完成的语气，不猜测后半句；保留时态、否定、疑问和人称，'
            '不要把叙述改成劝告。原文省略主语时，不擅自补出“我”或“你”。'
            '不使用文言文或古风措辞，不扩写，不解释。'
            '背景信息只用于理解指代，绝不混入本次译文。只输出译文：\n\n'
            + text
        )
        return [{'role': 'user', 'content': '\n\n'.join(sections)}]


# Apply only matched live-stream terms; custom glossary entries take precedence.
_LIVE_GLOSSARY = {
    "テンセント": "腾讯", "スパチャ": "SC打赏", "切り抜き": "直播切片",
    "推し変": "换推", "箱推し": "全团都推", "初見プレイ": "初次游玩",
    "初見詐欺": "假装新观众", "初見さん": "新观众", "指示厨": "指挥党",
    "鳩行為": "跨直播间传话", "杞憂民": "爱操心的观众",
    "案件配信": "商单直播", "同接": "同时在线人数", "古参ヅラ": "摆老粉架子",
    "親フラ": "家长突然出现", "サ終": "停服", "凸待ち": "等待连麦",
    "逆凸": "主动找人连麦", "飯テロ": "晒美食馋人", "台パン": "拍桌子",
    "積みゲー": "买了还没玩的游戏", "ROMって": "潜水围观",
    "ROM専": "只看不发言", "おま環": "个人设备或网络环境的问题",
    "鯖落ち": "服务器宕机", "スパチャ読み": "读SC打赏留言",
    "アーカイブ": "回放", "無凸": "未突破", "お局": "爱摆资历的大姐",
}


def _matched_live_terms(text: str) -> dict[str, str]:
    terms = {src: dst for src, dst in _LIVE_GLOSSARY.items() if src in text}
    # These words also have literal senses outside gacha games.
    if "ガチャ" in text:
        for src, dst in (("ガチャ", "抽卡"), ("天井", "保底"),
                         ("すり抜け", "抽卡歪了，抽到非当期目标")):
            if src in text:
                terms[src] = dst
    if "サ終" in text and "赤字" in text:
        terms["赤字"] = "亏损"
    if "ネタバレ" in text and "布教" in text:
        terms["布教"] = "安利推荐"
    return terms


class SakuraGalTransl(TranslateGemma):
    """日译中专用模型，保留其训练模板并适配直播口语。"""

    def __init__(self, base, *, temperature=0.0, repeat_penalty=1.0, seed=None):
        super().__init__(base)
        self.temperature = temperature
        self.repeat_penalty = repeat_penalty
        self.seed = seed

    def check_output(self, source, output, finish=None):
        if finish in ('length', 'empty', 'content_filter'):
            raise TranslationOutputError('本地翻译未正常完成')
        if '参考以下术语表' in output and '根据以上术语表' in output:
            raise TranslationOutputError('本地翻译复述了提示模板')
        if len(output) > max(96, len(source) * 2):
            repeated = re.search(r'(.{2,24})\1{5,}$', output[-300:], re.DOTALL)
            if repeated and set(repeated[1]) - set('哈呵嘿嘻啊呀哇！!？?。，,、…~～wW '):
                raise TranslationOutputError('本地翻译陷入重复生成')

    def messages(self, text, context, glossary):
        # Canonical spelling of FF game titles avoids the observed ff14 loop.
        text = re.sub(r'(?i)(?<![A-Za-z0-9])ff(\d{1,2})(?![A-Za-z0-9])',
                      lambda match: 'FF' + match[1], text)
        # Arabic time digits avoid observed kanji-hour copying errors.
        digits = {'〇': 0, '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                  '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
        def number_value(number):
            if '十' in number:
                tens, ones = number.split('十')
                value = (digits[tens] if tens else 1) * 10 + (digits[ones] if ones else 0)
            else:
                value = int(''.join(str(digits[c]) for c in number))
            return str(value)
        def time_digits(match):
            return (number_value(match[1]) + '時'
                    + (number_value(match[2]) + '分' if match[2] else '半'))
        number = r'([一二三四五六七八九]?十[〇零一二三四五六七八九]?|[〇零一二三四五六七八九]{1,2})'
        # Only complete clock expressions: standalone 十分 also means "enough".
        # Bare 一時 can mean "temporarily", so leave it unchanged too.
        pattern = (r'(?<![〇零一二三四五六七八九十百千万億兆数何幾\d])'
                   + number + r'時(?:' + number + r'分|半)')
        text = re.sub(pattern, time_digits, text)
        system = self.system_prompt()
        terms = dict((a, b) for a, b in glossary if a and b)
        for src, dst in _matched_live_terms(text).items():
            terms.setdefault(src, dst)
        background = [str(c).removeprefix('历史译文：').strip() for c in context
                      if isinstance(c, str) and c.startswith('历史译文：')]
        history = ('历史翻译：' + '\n'.join(background[-2:])[-240:] + '\n\n'
                   if background else '')
        user = (history + '参考以下术语表（可为空，格式为src->dst #备注）：\n'
                + '\n'.join(a + '->' + b for a, b in list(terms.items())[:40])
                + '\n\n根据以上术语表的对应关系和备注，结合历史剧情和上下文，'
                  '将下面的文本从日文翻译成简体中文：\n' + text)
        return [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]

    def system_prompt(self):
        return (
            '你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，'
            '并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，'
            '不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。'
            '本次内容是日语直播口语，请用自然、简洁的现代中文表达。'
            '省略主语时不要自行补出我、你、他或她；未完句不要猜测后半句。'
            '准确保留时间、数字、否定和疑问，只输出本次文本的译文。'
        )
    def sampling(self):
        values = {'temperature': self.temperature, 'top_p': 0.8,
                  'repeat_penalty': self.repeat_penalty, 'repeat_last_n': 128,
                  'chat_template_kwargs': {'enable_thinking': False}}
        if self.seed is not None:
            values['seed'] = self.seed
        return values


class OpenAICompat(LanguageProvider):
    kind = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0,
                 *, source_lang: str = "ja", target_lang: str = "zh"):
        super().__init__(source_lang=source_lang, target_lang=target_lang)
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout = timeout
        if not self.base_url or not self.model:
            raise RuntimeError("服务未配置：请填写 Base URL 与模型名")

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": build_messages(text, context, glossary, self.source_lang, self.target_lang),
            "temperature": 0.2,
            "max_tokens": 800,
            "stream": True,
        }
        parts: list[str] = []
        completed = False
        _checkpoint(on_delta)
        with make_client(self.base_url, timeout=_stream_timeout(on_delta, self.timeout)) as client:
            with client.stream("POST", f"{self.base_url}/chat/completions",
                               headers=headers, json=payload) as r:
                for line in _event_lines(r, on_delta):
                    _checkpoint(on_delta)
                    if not line:
                        continue
                    if line.startswith("data:"):
                        chunk = line[5:].strip()
                        if chunk == "[DONE]":
                            completed = True
                            break
                        try:
                            obj = json.loads(chunk)
                        except (ValueError, TypeError) as exc:
                            raise TranslationOutputError("翻译数据损坏，译文不完整") from exc
                        if not isinstance(obj, dict):
                            raise TranslationOutputError("翻译数据格式错误，译文不完整")
                        if obj.get("error"):
                            raise RuntimeError(str(obj["error"]))
                        choices = obj.get("choices") or []
                        if not choices:
                            continue  # usage-only events are valid SSE chunks
                        choice = choices[0]
                        if choice.get("finish_reason") in ("length", "content_filter"):
                            raise TranslationOutputError("翻译未正常完成，译文不完整")
                        if choice.get("finish_reason"):
                            completed = True
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            parts.append(piece)
                            if on_delta:
                                on_delta(piece)
                        if completed:
                            break  # Release the inference slot even if [DONE] never arrives.
        _checkpoint(on_delta)
        if not completed:
            raise TranslationOutputError("翻译连接提前结束，译文不完整")
        out = "".join(parts).strip().strip('"').strip()
        if not out:
            raise RuntimeError("模型没有返回内容")
        return out


class DeepL(LanguageProvider):
    kind = "deepl"

    def __init__(self, api_key: str, base_url: str = "", timeout: float = 15.0,
                 *, source_lang: str = "ja", target_lang: str = "zh"):
        super().__init__(source_lang=source_lang, target_lang=target_lang)
        self.api_key = api_key or ""
        if not self.api_key:
            raise RuntimeError("DeepL 需要填写 API Key")
        self.base_url = (base_url or ("https://api-free.deepl.com" if self.api_key.endswith(":fx")
                                      else "https://api.deepl.com")).rstrip("/")
        self.timeout = timeout

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:  # noqa: ARG002
        data = {"auth_key": self.api_key, "text": text,
                "source_lang": self.source_lang.upper(), "target_lang": self.target_lang.upper()}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/v2/translate", data=data)
            r.raise_for_status()
            return r.json()["translations"][0]["text"].strip()


class GoogleFree(LanguageProvider):
    kind = "googlefree"
    _URL = "https://translate.googleapis.com/translate_a/single"

    def __init__(self, timeout: float = 8.0, *, source_lang: str = "ja", target_lang: str = "zh"):
        super().__init__(source_lang=source_lang, target_lang=target_lang)
        self.timeout = timeout

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:  # noqa: ARG002
        params = {"client": "gtx", "sl": "zh-CN" if self.source_lang == "zh" else "ja",
                  "tl": "zh-CN" if self.target_lang == "zh" else "ja", "dt": "t", "q": text}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._URL, params=params)
            r.raise_for_status()
            data = r.json()
        return "".join(seg[0] for seg in data[0] if seg and seg[0]).strip()


class MicrosoftFree(LanguageProvider):
    kind = "msfree"
    _AUTH = "https://edge.microsoft.com/translate/auth"
    _API = "https://api-edge.cognitive.microsofttranslator.com/translate"
    _token = ""
    _token_at = 0.0

    def __init__(self, timeout: float = 8.0, *, source_lang: str = "ja", target_lang: str = "zh"):
        super().__init__(source_lang=source_lang, target_lang=target_lang)
        self.timeout = timeout

    def _get_token(self, client: httpx.Client) -> str:
        now = time.time()
        if MicrosoftFree._token and now - MicrosoftFree._token_at < 480:
            return MicrosoftFree._token
        r = client.get(MicrosoftFree._AUTH)
        r.raise_for_status()
        MicrosoftFree._token = r.text.strip()
        MicrosoftFree._token_at = now
        return MicrosoftFree._token

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:  # noqa: ARG002
        with httpx.Client(timeout=self.timeout) as client:
            token = self._get_token(client)
            params = {"api-version": "3.0", "from": "zh-Hans" if self.source_lang == "zh" else "ja",
                      "to": "zh-Hans" if self.target_lang == "zh" else "ja"}
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            r = client.post(self._API, params=params, headers=headers, json=[{"Text": text}])
            r.raise_for_status()
            return r.json()[0]["translations"][0]["text"].strip()


class MyMemoryFree(LanguageProvider):
    kind = "mymemory"
    _URL = "https://api.mymemory.translated.net/get"

    def __init__(self, timeout: float = 8.0, *, source_lang: str = "ja", target_lang: str = "zh"):
        super().__init__(source_lang=source_lang, target_lang=target_lang)
        self.timeout = timeout

    def translate(self, text: str, context: list[str], glossary, on_delta: OnDelta = None) -> str:  # noqa: ARG002
        source = "zh-CN" if self.source_lang == "zh" else "ja"
        target = "zh-CN" if self.target_lang == "zh" else "ja"
        params = {"q": text[:480], "langpair": source + "|" + target}
        with httpx.Client(timeout=self.timeout) as client:
            r = client.get(self._URL, params=params)
            r.raise_for_status()
            data = r.json()
        out = (data.get("responseData") or {}).get("translatedText") or ""
        if not out or "MYMEMORY WARNING" in out.upper():
            raise RuntimeError("MyMemory 额度限制")
        return out.strip()


_KINDS = {
    "openai": OpenAICompat,
    "deepl": DeepL,
    "googlefree": GoogleFree,
    "msfree": MicrosoftFree,
    "mymemory": MyMemoryFree,
}

FALLBACK_ORDER = ["ms-free", "mymemory-free", "google-free"]

# 本地服务可用性缓存：{base_url: (是否可用, 检查时间)}
_LOCAL_READY_CACHE: dict[str, tuple[bool, float]] = {}
_LOCAL_READY_TTL = 20.0
_LAST_GOOD_FALLBACK: list[str] = []  # 最近成功的免密钥通道，优先尝试
# 翻译服务实例缓存（避免每次翻译都重新构造 provider 对象）
# key: provider_id -> (provider实例, 构造时使用的 service_dict 哈希)
import hashlib as _hashlib
_PROVIDER_CACHE: dict[str, tuple[object, str]] = {}
_PROVIDER_CACHE_LOCK = __import__("threading").Lock()


def _service_sig(svc: dict) -> str:
    """把 service dict 的关键字段变成一个字符串签名，用于缓存失效判断。"""
    key = (f"{svc.get('kind')}|{svc.get('base_url')}|{svc.get('api_key')}|{svc.get('model')}|{svc.get('variant')}"
           f"|{svc.get('source_lang', 'ja')}|{svc.get('target_lang', 'zh')}")
    return key


def get_cached_provider(provider_id: str, svc: dict):
    """获取或新建翻译 provider 实例（线程安全，同一配置只构造一次）。"""
    sig = _service_sig(svc)
    with _PROVIDER_CACHE_LOCK:
        cached = _PROVIDER_CACHE.get(provider_id)
        if cached is not None and cached[1] == sig:
            return cached[0]
        inst = make_provider(svc)
        _PROVIDER_CACHE[provider_id] = (inst, sig)
        return inst


def _ollama_ready(base_url: str) -> bool:
    """快速检查 Ollama 是否在线（0.8 秒超时 + 结果缓存 20 秒）。"""
    base = base_url or "http://127.0.0.1:11434"
    now = time.time()
    hit = _LOCAL_READY_CACHE.get(base)
    if hit and now - hit[1] < _LOCAL_READY_TTL:
        return hit[0]
    from .ollama import ping
    try:
        ok = ping(base, timeout=0.8)
    except Exception:  # noqa: BLE001
        ok = False
    _LOCAL_READY_CACHE[base] = (ok, now)
    return ok


def _localmt_ready(source_lang: str = "ja", target_lang: str = "zh") -> bool:
    from .local_mt import installed
    return installed(source_lang, target_lang) or (installed(source_lang, "en") and installed("en", target_lang))


def make_provider(service: dict):
    kind = service.get("kind", "openai")
    languages = {"source_lang": service.get("source_lang", "ja"), "target_lang": service.get("target_lang", "zh")}
    if kind == "llamacpp":
        return LlamaCppProvider(service)
    if kind == "ollama":
        return OllamaNative(service.get("base_url", ""), service.get("model", ""), **languages)
    if kind == "localmt":
        return LocalMTProvider(**languages)
    if kind == "openai":
        return OpenAICompat(service.get("base_url", ""), service.get("api_key", ""), service.get("model", ""), **languages)
    if kind == "deepl":
        return DeepL(service.get("api_key", ""), service.get("base_url", ""), **languages)
    if kind == "googlefree":
        return GoogleFree(**languages)
    if kind == "msfree":
        return MicrosoftFree(**languages)
    if kind == "mymemory":
        return MyMemoryFree(**languages)
    raise RuntimeError(f"未知的翻译服务类型：{kind}")


class Translator:
    """带自动降级的翻译调度器：主服务 → 免密钥通道并发竞速（谁快用谁）。"""

    def __init__(self, config):
        self.config = config
        self.last_provider = ""
        self.last_error = ""
        self.last_engine = ""
        self.last_engine_reason = ""
        self.last_reason = ""     # 主服务被跳过/失败的原因（降级时给用户看）

    def _service(self, provider: str | None = None) -> dict:
        pid = provider or self.config.get("translate.provider", "ms-free")
        svc = self.config.service(pid)
        svc["source_lang"] = self.config.get("translate.source_lang", "ja")
        svc["target_lang"] = self.config.get("translate.target_lang", "zh")
        if svc.get("kind") == "llamacpp":
            # 内置引擎要知道用哪个变体（CPU / Vulkan / CUDA）——以前这里漏了，
            # 导致不管设置里选什么都只会用 CPU 引擎
            from .llama_engine import preferred_variant

            svc["variant"] = preferred_variant(self.config)
        return svc

    def _skip_reason(self, svc: dict) -> str | None:
        """判断某个服务现在是否可以直接跳过，返回原因文本。"""
        kind = svc.get("kind", "openai")
        base_url = svc.get("base_url", "") or ""
        local = "localhost" in base_url or "127.0.0.1" in base_url
        if kind == "ollama" and not _ollama_ready(base_url):
            return f"{svc.get('label', '')}：本地服务未运行（安装并启动 Ollama 后即可使用）"
        if kind == "localmt" and not _localmt_ready(svc.get("source_lang", "ja"), svc.get("target_lang", "zh")):
            return f"{svc.get('label', '')}：所选方向的内置翻译模型尚未安装"
        if kind == "llamacpp":
            from .llama_engine import installed_variants, resolve_model

            if not installed_variants():
                return f"{svc.get('label', '')}：内置引擎尚未下载（设置里点『下载引擎』）"
            if resolve_model(svc.get("model", "") or "") is None:
                return f"{svc.get('label', '')}：还没有 GGUF 模型（面板里可一键导入 Ollama 已下载的模型）"
        if kind == "openai" and not base_url:
            return f"{svc.get('label', '')}：缺少 Base URL"
        if kind in ("openai", "deepl") and not svc.get("api_key") and not local:
            return f"{svc.get('label', '')}：缺少 API Key"
        return None

    def _order_fallbacks(self, provider: str) -> list[str]:
        rest = [p for p in FALLBACK_ORDER if p != provider]
        if _LAST_GOOD_FALLBACK:
            top = [p for p in _LAST_GOOD_FALLBACK if p in rest]
            rest = top + [p for p in rest if p not in top]
        return rest

    def _race_keyless(self, text: str, context: list[str], glossary,
                      on_delta: OnDelta = None, provider: str | None = None) -> tuple[str, str, list[str]]:
        """几个免密钥通道并发翻译，谁先成功用谁。返回 (通道, 译文, 错误列表)。"""
        import queue as _queue
        import threading as _threading

        _checkpoint(on_delta)
        provider = provider or self.config.get("translate.provider", "ms-free")
        order = self._order_fallbacks(provider)
        q: _queue.Queue = _queue.Queue()
        errors: list[str] = []

        def work(pid: str) -> None:
            svc = self._service(pid)
            try:
                out = make_provider(svc).translate(text, context, glossary, None)
                q.put((pid, out or "", ""))
            except Exception as e:  # noqa: BLE001
                q.put((pid, "", f"{svc.get('label', pid)}：{_friendly_http_error(e)}"))

        for pid in order:
            _threading.Thread(target=work, args=(pid,), daemon=True).start()
        deadline = time.monotonic() + 12.0
        got = 0
        while got < len(order):
            _checkpoint(on_delta)
            remain = deadline - time.monotonic()
            if remain <= 0:
                break
            try:
                pid, out, err = q.get(timeout=min(remain, 0.1))
            except _queue.Empty:
                continue
            _checkpoint(on_delta)
            got += 1
            if out:
                if pid not in _LAST_GOOD_FALLBACK:
                    _LAST_GOOD_FALLBACK.insert(0, pid)
                    del _LAST_GOOD_FALLBACK[3:]
                return pid, out, errors
            if err:
                errors.append(err)
        return "", "", errors

    def translate(self, text: str, context: list[str], on_delta: OnDelta = None,
                  provider: str | None = None) -> str:
        """翻译文本；主服务失败或未就绪时自动改用免密钥通道。

        provider 参数可临时指定本次使用的服务（例如草稿译文走极速模型）。
        """
        text = (text or "").strip()
        if not text:
            return ""
        _checkpoint(on_delta)
        self.last_provider = self.last_error = self.last_engine = ""
        self.last_engine_reason = self.last_reason = ""
        glossary = [tuple(x) for x in (self.config.get("translate.glossary", []) or [])]
        provider = provider or self.config.get("translate.provider", "ms-free")
        errors: list[str] = []

        svc = self._service(provider)
        if (svc["source_lang"], svc["target_lang"]) == ("zh", "ja"):
            glossary = [(target, source) for source, target in glossary]
        reason = self._skip_reason(svc)
        if reason:
            errors.append(reason)
        else:
            try:
                instance = get_cached_provider(provider, svc)
                out = instance.translate(text, context, glossary, on_delta)
                _checkpoint(on_delta)
                self.last_engine = getattr(instance, "actual_variant", "")
                self.last_engine_reason = getattr(instance, "engine_reason", "")
                if not out or not out.strip():
                    raise RuntimeError("翻译服务返回空内容")
                self.last_provider = provider
                self.last_error = ""
                return out
            except (TranslationCancelled, TranslationDeadlineExceeded, TranslationOutputError):
                raise
            except Exception as e:  # noqa: BLE001
                errors.append(f"{svc.get('label', provider)}：{_friendly_http_error(e)}")

        self.last_reason = errors[0] if errors else ""
        if not self.config.get("translate.auto_fallback", False):
            self.last_error = "；".join(errors)
            raise RuntimeError(self.last_error)
        pid, out, errs = self._race_keyless(text, context, glossary, on_delta, provider)
        errors.extend(errs)
        if out:
            self.last_provider = pid
            self.last_error = ""
            return out
        self.last_error = "；".join(errors) or "没有可用的翻译服务"
        raise RuntimeError(self.last_error)

    def test(self, provider: Optional[str] = None) -> str:
        """测试某个服务是否可用，返回中文结果说明。"""
        svc = self._service(provider)
        try:
            text = "你好，这是测试。" if svc.get("source_lang") == "zh" else "こんにちは、テストです。"
            out = make_provider(svc).translate(text, [], [])
            return f"连接成功：{out}"
        except Exception as e:  # noqa: BLE001
            return f"连接失败：{_friendly_http_error(e)}"

