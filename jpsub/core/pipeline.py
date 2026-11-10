# -*- coding: utf-8 -*-
"""主流水线：音频采集 → 静音检测断句 → 流式出字 → 整句高精度校对 → 翻译。"""
from __future__ import annotations

import collections
import json
import logging
from logging.handlers import QueueListener, RotatingFileHandler
import math
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6 import QtCore

from ..paths import MODELS, USER_DATA
from .asr import FinalASR
from .audio import AudioCapture, normalize_gain
from .readiness import live_provider_for, vad_ready
from .segment import Segment
from .translate import (Translator, TranslationCancelled, TranslationStream,
                        TranslationOutputError, TranslationDeadlineExceeded)

# 草稿重识别最多占用这么多比例的墙钟时间，剩下的留给整句校对和翻译；
# 一句话越长，重识别一次越慢，草稿就会自动跟着放慢（否则会越跑越卡）。
DRAFT_CPU_BUDGET = 0.35
# 实时草稿译文每跑 1 秒，至少留这么多秒给整句翻译，避免本地模型被草稿占满。
LIVE_DUTY_GAP = 1.2
# These bounds are for live listening, not offline batch transcription. Never
# let a slow model turn a live stream into an ever-growing FIFO replay.
MAX_PENDING_FINALS = 8
MAX_TRANSLATION_AGE_S = 12.0
MAX_ASR_QUEUE_AGE_S = 12.0
PERFORMANCE_INTERVAL_S = 30.0


class _PerformanceFileHandler(RotatingFileHandler):
    """Best-effort diagnostics: disk failures must not leak into subtitle work."""

    def _open(self):
        # This method runs in the listener, never the audio/ASR/translation worker.
        from pathlib import Path
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()

    def handleError(self, record):
        pass


class _PerformanceWriter:
    def __init__(self, path):
        self.queue = queue.Queue(maxsize=16)
        self.handler = _PerformanceFileHandler(
            path, maxBytes=512 * 1024, backupCount=2, encoding="utf-8", delay=True)
        self.handler.setLevel(logging.INFO)
        self.handler.setFormatter(logging.Formatter("%(message)s"))
        self.listener = QueueListener(self.queue, self.handler)
        self.listener.start()

    def submit(self, metrics):
        # Bounded and non-blocking, including when the filesystem stalls.
        record = logging.LogRecord("jpsub.performance", logging.INFO, "", 0,
                                   json.dumps(metrics, separators=(",", ":")), (), None)
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass


_performance_writer = None
_performance_writer_lock = threading.Lock()


def _write_performance(metrics):
    """One listener/rotating file per process, shared across listening sessions."""
    global _performance_writer
    try:
        if _performance_writer is None:
            if not _performance_writer_lock.acquire(blocking=False):
                return
            try:
                if _performance_writer is None:
                    _performance_writer = _PerformanceWriter(USER_DATA / "performance.log")
            finally:
                _performance_writer_lock.release()
        _performance_writer.submit(metrics)
    except Exception:  # Diagnostics are optional, including thread/resource failures.
        pass


class Pipeline(QtCore.QThread):
    """在后台线程中运行整条识别翻译流水线。"""

    sig_status = QtCore.Signal(str)
    sig_error = QtCore.Signal(str)
    sig_partial = QtCore.Signal(str)             # 流式日语文本（空串=清空）
    sig_segment = QtCore.Signal(object)          # Segment 最终日语
    sig_live_zh = QtCore.Signal(str, str)        # (sid, 实时译文增量后的全文)
    sig_final_zh = QtCore.Signal(str, str)       # (sid, 最终译文)
    sig_translate_queue = QtCore.Signal(dict)
    sig_translate_failed = QtCore.Signal(str, str)
    sig_translate_info = QtCore.Signal(dict)     # 本次翻译来自哪个服务/耗时
    sig_level = QtCore.Signal(float)             # 0~1 音量
    sig_stats = QtCore.Signal(dict)              # 统计信息

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.cfg = config
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._reset_pending = threading.Event()
        block_ms = max(10, int(config.get("audio.block_ms", 80)))
        self._queue: queue.Queue = queue.Queue(maxsize=max(2, 2000 // block_ms))
        self._captured_samples = 0
        self._audio_next_position = 0
        self._sample_zero_at: float | None = None
        # One capture-thread writer publishes immutable snapshots. These are
        # observations of delivery, independent of playback and subtitle clocks.
        self._capture_clock: tuple[float, float, int] | None = None
        self._audio_overload_totals = (0, 0)  # events, discarded 16 kHz samples
        self._capture: AudioCapture | None = None
        self._final_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="final-asr")
        self._draft_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="draft-asr")
        self._trans_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translate")
        self._live_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translate-live")
        self._pools_closed = False
        self._translate_gate = threading.Lock()
        self._translation_state_lock = threading.Lock()
        self._pending_final = {}
        self._translation_jobs = collections.OrderedDict()
        self._translation_worker_running = False
        self._asr_state_lock = threading.Lock()
        self._asr_jobs = collections.OrderedDict()
        self._asr_worker_running = False
        self._failed_translations = {}
        self._live_task_inflight = False
        self._preview_lock = threading.RLock()
        self._pending_preview = None  # One replaceable (generation, revision sid, source, arrival time).
        self._preview_revision = 0
        self._last_queue_report = 0.0
        self._performance_started = time.monotonic()
        self._last_performance_report = self._performance_started
        self._performance_report_lock = threading.Lock()

        # 运行状态
        self._fed = 0                    # 已处理样本数（16k）
        self._pre_roll: collections.deque = collections.deque(maxlen=18)   # ~0.56s 预卷
        self._raw_frames: collections.deque = collections.deque(maxlen=1250)  # 40 秒原始音频
        self._vad_origin = 0
        self._active = False
        self._utterance: list[np.ndarray] = []
        self._utterance_start = 0.0
        self._partial = ""
        self._last_final_jp: list[str] = []
        self._last_final_zh: list[str] = []
        self._live = {"gen": 0, "sid": "", "busy": False, "last_req": 0.0, "text": "", "latency": 0.0}
        self._fallback_notified = ""
        self._stream = None
        self._draft_engine: FinalASR | None = None
        self._draft_mode = "off"
        self._draft_busy = False
        self._draft_started = 0.0
        self._draft_next_ok = 0.0
        self._final_busy = 0
        self._final: FinalASR | None = None
        self._vad = None
        self._level_counter = 0          # sig_level 降采样计数器
        self._max_segment_samples = 0   # 在 _setup 里初始化，避免热路径每帧读配置
        self._vad_pending = np.zeros(0, dtype=np.float32)
        self._vad_window = 512
        self._queue_warned = False

    # ------------------------------------------------------------- 生命周期
    def _submit(self, pool, fn, *args):
        """往线程池投任务。线程池已经关了就当没这回事——停止瞬间的收尾任务丢掉没关系。"""
        if self._pools_closed:
            return
        try:
            return pool.submit(fn, *args)
        except RuntimeError:      # 池已关闭
            pass

    def _shutdown_pools(self) -> None:
        if self._pools_closed:
            return
        self._pools_closed = True
        for pool in (self._final_pool, self._draft_pool, self._trans_pool, self._live_pool):
            try:
                pool.shutdown(wait=True, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass

    def _release_resources(self) -> None:
        """Called only after capture and model workers have stopped using them."""
        models = (self._final, self._draft_engine)
        self._final = self._draft_engine = None
        seen = set()
        for model in models:
            if model is not None and id(model) not in seen:
                seen.add(id(model))
                model.close()
        # Capture callbacks point back to this Pipeline; break that reference ring.
        self._capture = None
        self._vad = self._stream = None
        self._utterance.clear()
        self._pre_roll.clear()
        self._raw_frames.clear()
        self._vad_pending = np.zeros(0, dtype=np.float32)
        with self._translation_state_lock:
            self._translation_jobs.clear()
            self._pending_final.clear()
            self._translation_worker_running = False
        with self._asr_state_lock:
            self._asr_jobs.clear()
            self._asr_worker_running = False
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self, max_wait: float = 1.5) -> None:
        """请求停止。默认不等太久，避免界面在「停止 / 切换设置」时卡住。

        真正的收尾（关线程池、放掉模型）在流水线线程自己退出时做，所以哪怕它还在
        加载模型也不会把任务投进已经关掉的线程池里。
        """
        self._stop.set()
        if self._capture:
            self._capture.stop()
        self.wait(int(max_wait * 1000))

    def pause(self, flag: bool) -> None:
        if flag:
            self._pause.set()
            # Invalidate in-flight previews immediately. A quick pause/resume
            # can finish before the next audio chunk resets the utterance.
            with self._preview_lock:
                self._live["gen"] += 1
                self._live["sid"] = ""
                self._pending_preview = None
        else:
            # Do not replay queued audio from before/during the pause.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._reset_pending.set()
            self._pause.clear()

    # ------------------------------------------------------------- 主循环
    def run(self) -> None:  # noqa: C901
        try:
            try:
                self._setup()
            except Exception as e:  # noqa: BLE001
                self.sig_error.emit(str(e))
                self.sig_status.emit("启动失败")
                return

            if self._stop.is_set():
                return
            self.sig_status.emit("正在聆听…")
            while not self._stop.is_set():
                try:
                    chunk, rms, position = self._queue.get(timeout=0.2)
                except queue.Empty:
                    if self._pending_preview is not None:
                        self._maybe_live_translate(time.monotonic())
                    self._report_translation_queue()
                    continue
                if self._stop.is_set():
                    break
                if self._pause.is_set():
                    continue
                discontinuity = position != self._audio_next_position
                self._audio_next_position = position + chunk.size
                if self._reset_pending.is_set() or discontinuity:
                    self._reset_pending.clear()
                    self._pre_roll.clear()
                    self._reset_utterance()
                    self._vad_pending = np.zeros(0, dtype=np.float32)
                    self._vad.reset()
                    self._fed = position
                    self._vad_origin = self._fed
                    self._raw_frames.clear()
                try:
                    self._process_chunk(chunk, rms)
                    self._report_translation_queue()
                except Exception as e:  # noqa: BLE001
                    self.sig_error.emit(f"处理音频出错：{e}")

            # 收尾：如果还有未完成的语音，做最后一次校对
            if self._active and self._utterance:
                self._finalize_utterance()
            self.sig_level.emit(0.0)
        finally:
            if self._capture:
                self._capture.stop()
                self._capture.join(timeout=2.0)
            # finished 代表所有模型工作也已结束。
            self._shutdown_pools()
            self._release_resources()

    # ------------------------------------------------------------- 初始化
    def _setup(self) -> None:
        cfg = self.cfg
        # 0) 先定下整句校对引擎：草稿要不要和它共用同一份模型实例，取决于这里
        engine = "parakeet"
        cloud = {}
        if not FinalASR.available(engine):
            raise FileNotFoundError("日语识别模型不完整，请重新安装软件")
        vad_path = MODELS / "silero_vad.onnx"
        if not vad_ready(vad_path):
            raise FileNotFoundError("静音检测模型 silero_vad.onnx 缺失或不完整，请用完整发行包修复安装。")
        final_threads = int(cfg.get("asr.final_threads", 6))
        # 1) 实时出字：用唯一模型滚动识别
        want_mode = str(cfg.get("asr.draft_mode", "rolling"))
        if not bool(cfg.get("asr.partial", True)):
            want_mode = "off"
        self._draft_mode = "off"
        if want_mode != "off":
            self.sig_status.emit("正在加载日语识别模型…")
            # 草稿和整句校对共用唯一模型和原生服务进程。
            self._draft_engine = FinalASR("parakeet", num_threads=final_threads)
            self._draft_engine.load()
            self._draft_mode = "rolling"
        # 2) 静音检测
        import sherpa_onnx

        vcfg = sherpa_onnx.VadModelConfig()
        vcfg.silero_vad.model = str(vad_path)
        vcfg.silero_vad.window_size = self._vad_window
        vcfg.silero_vad.threshold = float(cfg.get("vad.threshold", 0.5))
        vcfg.silero_vad.min_silence_duration = int(cfg.get("vad.min_silence_ms", 350)) / 1000.0
        vcfg.silero_vad.min_speech_duration = int(cfg.get("vad.min_speech_ms", 200)) / 1000.0
        vcfg.silero_vad.max_speech_duration = float(cfg.get("asr.max_segment_s", 24))
        vcfg.sample_rate = 16000
        vcfg.num_threads = 1
        self._vad = sherpa_onnx.VoiceActivityDetector(vcfg, buffer_size_in_seconds=60)
        # 3) 整句校对引擎（异步预加载）
        if (self._draft_mode == "rolling" and self._draft_engine is not None
                and self._same_engine(self._draft_engine.engine, engine)):
            # 共用上面已经加载好的那一份（草稿 ≠ 校对只是两个用途，模型是同一个）
            self._final = self._draft_engine
        else:
            self._final = FinalASR(engine, num_threads=final_threads, cloud=cloud)
        self._preload_final()
        if self._stop.is_set():
            return
        if cfg.get("translate.enabled", True) and cfg.get("translate.provider", "") == "llamacpp":
            self.sig_status.emit("正在加载本地翻译模型…")
            self._warmup_translate()
            if self._stop.is_set():
                return
        # 4) 音频采集
        self.sig_status.emit("正在打开音频设备…")
        self._capture = AudioCapture(
            source=str(cfg.get("audio.source", "loopback")),
            device_name=str(cfg.get("audio.device", "") or ""),
            on_chunk=self._on_audio,
            on_error=self._on_capture_error,
            block_ms=int(cfg.get("audio.block_ms", 80)),
            on_status=lambda m: self.sig_status.emit(m),
        )
        self._fed = 0
        self._captured_samples = self._audio_next_position = 0
        self._capture.start()
        self._max_segment_samples = int(cfg.get("asr.max_segment_s", 24)) * 1000 // 80

    @staticmethod
    def _same_engine(a: str, b: str) -> bool:
        """两个角色能不能共用同一份模型实例：同名、而且是 sherpa 系离线模型才行。"""
        if a != b:
            return False
        from .config import FINAL_ENGINES

        return str(FINAL_ENGINES.get(b, {}).get("kind") or "sherpa") in ("sherpa", "native")

    def _preload_final(self) -> None:
        try:
            self._final.load()
            self.sig_status.emit("模型就绪，正在聆听…")
        except Exception as e:  # noqa: BLE001
            self.sig_error.emit(f"校对模型加载失败：{e}")
            raise

    # ------------------------------------------------------------- 音频处理
    def _on_capture_error(self, message: str) -> None:
        # AudioCapture exits after a device error. End its owning pipeline too,
        # allowing the UI to restart instead of displaying a dead listening job.
        self._stop.set()
        self.sig_error.emit(message)
        self.sig_status.emit("音频采集失败，请检查设备后重新开始")

    @property
    def subtitle_audio_clock(self):
        """Approximate monotonic time of sample zero, and delivered subtitle audio seconds."""
        return self._sample_zero_at, self._captured_samples / 16000.0

    def _on_audio(self, chunk: np.ndarray, rms: float) -> None:
        if self._stop.is_set():
            return
        if chunk.size:
            arrived = time.monotonic()
            clock = self._capture_clock
            # First callback marks the end of its audio; only subsequent
            # samples span the measured interval. Count deliveries even while
            # paused, without advancing the ASR timeline or enqueuing them.
            self._capture_clock = ((arrived, arrived, 0) if clock is None else
                                   (clock[0], arrived, clock[2] + chunk.size))
        if self._pause.is_set():
            return
        if chunk.size and self._sample_zero_at is None:
            # Callbacks arrive at the end of a block. Exclude model/device warmup.
            self._sample_zero_at = arrived - chunk.size / 16000.0
        position = self._captured_samples
        self._captured_samples += chunk.size
        try:
            self._queue.put_nowait((chunk, rms, position))
        except queue.Full:
            # Discard old buffered audio, not the newest live audio. Position
            # tags make the consumer reset VAD rather than splice unrelated
            # syllables together or compress the subtitle timeline.
            skipped = 0
            while True:
                try:
                    old, _, _ = self._queue.get_nowait()
                    skipped += old.size
                except queue.Empty:
                    break
            self._queue.put_nowait((chunk, rms, position))
            if skipped:
                events, dropped = self._audio_overload_totals
                self._audio_overload_totals = (events + 1, dropped + skipped)
            if not self._queue_warned:
                self._queue_warned = True
                self.sig_error.emit("音频处理过载，已跳过积压音频以跟上直播；可关闭实时草稿或减少后台程序")
            self.sig_status.emit(f"音频过载：跳过 {skipped / 16000:.1f} 秒积压音频，正在恢复实时聆听")

    def _process_chunk(self, chunk: np.ndarray, rms: float) -> None:
        level = min(1.0, max(0.0, (20 * np.log10(rms + 1e-6) + 60) / 60)) if rms > 1e-6 else 0.0
        self._level_counter += 1
        if self._level_counter % 4 == 0:
            self.sig_level.emit(level)
        self._vad_pending = np.concatenate((self._vad_pending, chunk))
        while self._vad_pending.size >= self._vad_window:
            frame = self._vad_pending[:self._vad_window]
            self._vad_pending = self._vad_pending[self._vad_window:]
            self._process_frame(frame)
        # ASR need not produce another *different* draft to wake a throttled
        # preview. The existing capture loop is the clock; no timer per token.
        if self._pending_preview is not None:
            self._maybe_live_translate(time.monotonic())

    def _process_frame(self, frame: np.ndarray) -> None:
        now = time.monotonic()
        self._fed += frame.size
        self._raw_frames.append(frame.copy())
        # VAD 以固定窗口运行。整句必须使用 VAD 保存的音频，不能把队列直接 pop 掉。
        self._vad.accept_waveform(normalize_gain(frame, max_gain=6.0))
        speech = bool(self._vad.is_speech_detected())
        if speech and not self._active:
            self._active = True
            self._utterance = list(self._pre_roll)
            self._utterance_start = (self._fed - frame.size - sum(c.size for c in self._pre_roll)) / 16000
            with self._preview_lock:
                self._live['gen'] += 1
                self._live.update(sid=f"L{self._live['gen']}", busy=False, last_req=0.0, text='')
                self._pending_preview = None
            self._draft_next_ok = now + int(self.cfg.get('asr.draft_interval_ms', 900)) / 1000
        if self._active:
            self._utterance.append(frame)
        self._pre_roll.append(frame)
        if self._draft_mode == 'streaming' and self._stream is not None:
            text = self._stream.feed(frame)
            if text is not None and text != self._partial:
                self._publish_preview_source(self._live['gen'], text)
        while not self._vad.empty():
            segment = self._vad.front
            # VAD 接收的逐帧增益音频只用于检测；识别必须使用原始录音。
            # 否则每 32ms 一次的独立放大会改变音色，并放大停顿里的背景噪声。
            history_start = self._fed - sum(frame.size for frame in self._raw_frames)
            segment_start = self._vad_origin + segment.start
            begin = max(history_start, segment_start - 6400)
            end = min(self._fed, segment_start + len(segment.samples) + 2560)
            # Copy only this utterance, rather than concatenating the entire
            # 40-second history and then copying the selected range again.
            selected = []
            cursor = history_start
            for raw in self._raw_frames:
                next_cursor = cursor + raw.size
                if next_cursor > begin and cursor < end:
                    selected.append(raw[max(0, begin-cursor):min(raw.size, end-cursor)])
                cursor = next_cursor
                if cursor >= end:
                    break
            audio = np.concatenate(selected) if selected else np.zeros(0, dtype=np.float32)
            self._vad.pop()
            # VAD already waited for the configured silence. Submit immediately:
            # waiting for another utterance delays completed subtitles by seconds.
            self._finalize_utterance(audio=audio, start=begin / 16000.0)
        if self._active:
            self._maybe_rolling_draft(now)

    def _reset_utterance(self, keep_active: bool = False) -> str:
        with self._preview_lock:
            live_text = self._partial
            self._utterance = []
            self._active = keep_active
            self._partial = ''
            self._live['gen'] += 1
            self._live['sid'] = ''
            self._pending_preview = None
            self.sig_partial.emit('')
        self._draft_next_ok = 0.0
        if self._stream is not None:
            self._stream.reset()
        return live_text

    def _finalize_utterance(self, keep_active: bool = False,
                            audio: np.ndarray | None = None, start: float | None = None) -> None:
        if audio is None:
            audio = np.concatenate(self._utterance) if self._utterance else np.zeros(0, dtype=np.float32)
        live_text = self._reset_utterance(keep_active)
        t0 = self._utterance_start if start is None else start
        self._submit_final_audio(audio, t0, live_text)

    def _submit_final_audio(self, audio: np.ndarray, start: float, live_text: str) -> None:
        if audio.size < 4000 or self._stop.is_set():
            return
        seg = Segment(sid=uuid.uuid4().hex[:8], jp='', t0=start,
                      t1=start + audio.size / 16000.0, live_jp=live_text)
        with self._asr_state_lock:
            skipped = None
            if len(self._asr_jobs) >= MAX_PENDING_FINALS:
                _, skipped = self._asr_jobs.popitem(last=False)
            self._asr_jobs[seg.sid] = (seg, audio, time.monotonic())
            start_worker = not self._asr_worker_running
            self._asr_worker_running = True
        if skipped is not None:
            self._report_asr_overload(skipped[0])
        if start_worker and self._submit(self._final_pool, self._drain_final_audio) is None:
            with self._asr_state_lock:
                self._asr_jobs.clear()
                self._asr_worker_running = False

    def _report_asr_overload(self, seg):
        self.sig_error.emit(f"识别过载：跳过 {seg.t0:.1f}–{seg.t1:.1f} 秒的积压音频；该段未完成识别")

    def _drain_final_audio(self):
        # Only this one task is ever submitted to the executor. Bounding a
        # side dictionary while still submitting one Future per item would
        # leave the executor's private unbounded queue growing indefinitely.
        while True:
            with self._asr_state_lock:
                if self._stop.is_set() or not self._asr_jobs:
                    self._asr_jobs.clear()
                    self._asr_worker_running = False
                    return
                _, (seg, audio, queued_at) = self._asr_jobs.popitem(last=False)
            try:
                if time.monotonic() - queued_at > MAX_ASR_QUEUE_AGE_S:
                    self._report_asr_overload(seg)
                    continue
                self._run_final(seg, audio)
            except Exception as exc:
                # Post-decode processing can fail too. Keep draining so one
                # bad item cannot strand the running flag and disable ASR.
                self.sig_error.emit(f"整句识别任务失败：{exc}")

    def _run_final(self, seg: Segment, audio: np.ndarray) -> None:
        if self._stop.is_set():
            return
        self._final_busy += 1
        try:
            try:
                context = ("\n".join(self._last_final_jp[-3:])
                           if self.cfg.get("asr.context_prompt", True) else "")
                result = self._final.transcribe(audio, context)
            except Exception as e:  # noqa: BLE001
                self.sig_error.emit(f"整句识别失败：{e}")
                return
            self._emit_final(seg, result)
        finally:
            self._final_busy -= 1

    def _emit_final(self, seg: Segment, result) -> None:
        text = result.text.strip()
        if self._stop.is_set():
            return
        if not text:
            # The capture-side boundary already cleared this utterance.
            # A slow empty decode must not erase the next utterance's preview.
            return
        seg.jp = text
        seg.asr_seconds = result.seconds
        seg.rtf = result.rtf
        context = self._context_lines()
        self._last_final_jp.append(text)
        if len(self._last_final_jp) > 8:
            self._last_final_jp = self._last_final_jp[-8:]
        self.sig_segment.emit(seg)
        self.sig_stats.emit({"rtf": round(result.rtf, 2), "asr_ms": int(result.seconds * 1000),
                             "audio_s": round(result.audio_seconds, 1)})
        if self.cfg.get("translate.enabled", True):
            sid = seg.sid
            self.enqueue_translation(sid, text, context)

    # ------------------------------------------------------------- 滚动草稿
    def _maybe_rolling_draft(self, now: float) -> None:
        if self._draft_mode != "rolling" or self._draft_engine is None or self._draft_busy:
            return
        if self._final_busy or self._asr_worker_running:
            return          # 整句校对才是要用的结果，草稿先让路
        if now < self._draft_next_ok:
            return
        total = sum(len(c) for c in self._utterance)
        if total < int(16000 * 0.7):
            return
        self._draft_busy = True
        self._draft_started = now
        gen = self._live["gen"]
        audio = np.concatenate(self._utterance)
        if self._submit(self._draft_pool, self._run_draft, gen, audio) is None:
            self._draft_busy = False

    def _run_draft(self, gen: int, audio: np.ndarray) -> None:
        text = ""
        try:
            if (self._stop.is_set() or self._pause.is_set() or not self._active
                    or gen != self._live["gen"] or self._final_busy or self._asr_worker_running):
                return
            result = self._draft_engine.transcribe(audio)
            text = result.text.strip()
        except Exception:  # noqa: BLE001
            text = ""
        finally:
            self._draft_busy = False
            # 这一趟花了多久，下一趟就按比例多等一会儿：草稿最多吃掉 DRAFT_CPU_BUDGET 的时间。
            # （以前是固定 0.6 秒一次，一句话越长重识别越慢，最后会把 CPU 全占满）
            spent = max(0.0, time.monotonic() - self._draft_started)
            interval = int(self.cfg.get("asr.draft_interval_ms", 600)) / 1000.0
            self._draft_next_ok = self._draft_started + max(interval, spent / DRAFT_CPU_BUDGET)
        if (not text or self._stop.is_set() or self._pause.is_set()
                or gen != self._live["gen"] or not self._active or self._final_busy
                or self._asr_worker_running):
            return
        self._publish_preview_source(gen, text)

    # ------------------------------------------------------------- 翻译
    def _context_lines(self) -> list[str]:
        n = int(self.cfg.get("translate.context_lines", 3))
        if self._uses_sakura():
            return ['历史译文：' + text for text in self._last_final_zh[-n:]] if n > 0 else []
        return self._last_final_jp[-n:] if n > 0 else []

    def _uses_sakura(self) -> bool:
        model = str(self.cfg.service('llamacpp').get('model') or '').lower()
        return (self.cfg.get('translate.provider', '') == 'llamacpp'
                and 'galtransl' in model)

    def _warmup_translate(self) -> None:
        """启动时静默预热翻译服务（本地大模型先加载好，第一句就不用等）。"""
        try:
            translator = Translator(self.cfg)
            control = TranslationStream(lambda piece: None, self._stop.is_set,
                                        seconds=90.0, read_timeout=15.0)
            control.checkpoint()
            translator.translate("こんにちは。", [], on_delta=control)
            control.checkpoint()
            self.sig_translate_info.emit({"provider": translator.last_provider,
                                          "engine": translator.last_engine,
                                          "engine_reason": translator.last_engine_reason, "warmup": True})
        except Exception:  # noqa: BLE001
            pass

    def _report_translation_queue(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_queue_report < 1.0:
            return
        self._last_queue_report = now
        with self._translation_state_lock:
            pending = len(self._pending_final)
            oldest = max(0.0, now - min(self._pending_final.values())) if pending else 0.0
            failed = len(self._failed_translations)
        with self._asr_state_lock:
            asr_pending = len(self._asr_jobs) + max(0, self._final_busy)
        self.sig_translate_queue.emit({
            "pending": pending, "oldest_s": round(oldest), "failed": failed,
            "audio_pending": self._queue.qsize(), "asr_pending": asr_pending,
        })
        self._report_performance(now, pending, oldest, failed)

    def _report_performance(self, now, pending, oldest, failed):
        # This callback is shared by several workers. Skip a sample instead of
        # waiting for telemetry locks; no filesystem work occurs on this thread.
        # Compare against the absolute scheduled instant. Subtracting two
        # monotonic floats can round a due interval just below 30 seconds.
        if now < self._last_performance_report + PERFORMANCE_INTERVAL_S:
            return
        if not self._performance_report_lock.acquire(blocking=False):
            return
        try:
            if now < self._last_performance_report + PERFORMANCE_INTERVAL_S:
                return
            self._last_performance_report = now
            if not self._queue.mutex.acquire(blocking=False):
                return
            try:
                audio_count = len(self._queue.queue)
                # Buffered audio duration, NOT wall-clock delay or VAD sentence age.
                audio_buffered = sum(item[0].size for item in self._queue.queue) / 16000.0
            finally:
                self._queue.mutex.release()
            if not self._asr_state_lock.acquire(blocking=False):
                return
            try:
                asr_queued = len(self._asr_jobs)
                asr_running = int(self._final_busy > 0)
            finally:
                self._asr_state_lock.release()
            clock = self._capture_clock
            events, dropped = self._audio_overload_totals
            metrics = {
                "time_epoch_s": round(time.time(), 3),
                "elapsed_s": round(max(0.0, now - self._performance_started), 3),
                "audio_queue": audio_count,
                "audio_buffered_s": round(audio_buffered, 3),
                "audio_overload_events": events,
                "audio_overload_dropped_samples": dropped,
                "asr_queued": asr_queued, "asr_running": asr_running,
                "translation_pending": pending,
                "translation_oldest_s": round(oldest, 3),
                "translation_failed": failed,
            }
            if clock is not None:
                first, last, samples = clock
                metrics["capture_callback_age_s"] = round(max(0.0, now - last), 3)
                if samples:
                    # Positive: less audio arrived than elapsed wall time;
                    # negative: burst delivery. Neither is source/driver delay:
                    # a fixed upstream offset is invisible to this measure.
                    metrics["capture_clock_gap_s"] = round(last - first - samples / 16000.0, 3)
            _write_performance(metrics)
        except Exception:
            pass
        finally:
            self._performance_report_lock.release()

    def _has_final_pending(self):
        with self._translation_state_lock:
            return bool(self._pending_final)

    def _translation_age_limit(self):
        value = self.cfg.get("translate.max_latency_s", MAX_TRANSLATION_AGE_S)
        try:
            seconds = float(value)
            if isinstance(value, bool) or not math.isfinite(seconds):
                raise ValueError("invalid latency limit")
        except (TypeError, ValueError, OverflowError):
            seconds = MAX_TRANSLATION_AGE_S
        return max(2.0, min(60.0, seconds))

    def _translation_failed(self, sid, message, *, notify=False):
        with self._translation_state_lock:
            self._failed_translations[sid] = message
        self.sig_translate_failed.emit(sid, message)
        if notify:
            self.sig_error.emit(f"翻译失败（可点补译）：{message}")

    def enqueue_translation(self, sid, text, context=None, *, retry=False):
        if self._stop.is_set() or self._pools_closed or not text.strip():
            return False
        now = time.monotonic()
        expired = []
        with self._translation_state_lock:
            if sid in self._pending_final:
                return False
            # Bulk manual retries must not evict one another or current speech.
            if retry and len(self._pending_final) >= MAX_PENDING_FINALS:
                return False
            for old_sid in list(self._translation_jobs):
                if now - self._pending_final[old_sid] >= self._translation_age_limit():
                    del self._translation_jobs[old_sid]
                    del self._pending_final[old_sid]
                    expired.append(old_sid)
            while len(self._pending_final) >= MAX_PENDING_FINALS and self._translation_jobs:
                old_sid, _ = self._translation_jobs.popitem(last=False)
                del self._pending_final[old_sid]
                expired.append(old_sid)
            self._pending_final[sid] = now
            self._translation_jobs[sid] = (text, context)
            start_worker = not self._translation_worker_running
            self._translation_worker_running = True
        for old_sid in expired:
            self._translation_failed(old_sid, "翻译积压，已暂缓旧句以跟上直播；原文已保留，可点补译")
        self._report_translation_queue(True)
        if start_worker and self._submit(self._trans_pool, self._drain_translations) is None:
            with self._translation_state_lock:
                self._translation_jobs.clear()
                self._pending_final.clear()
                self._translation_worker_running = False
            self._report_translation_queue(True)
            return False
        return True

    def _drain_translations(self):
        while True:
            with self._translation_state_lock:
                if self._stop.is_set() or not self._translation_jobs:
                    self._translation_jobs.clear()
                    self._pending_final.clear()
                    self._translation_worker_running = False
                    return
                sid, (text, context) = self._translation_jobs.popitem(last=False)
            try:
                self._run_translate_final(sid, text, context)
            except Exception as exc:  # A bad item must not strand the drain flag.
                self._translation_failed(sid, str(exc), notify=True)
            finally:
                with self._translation_state_lock:
                    self._pending_final.pop(sid, None)
            self._report_translation_queue(True)

    def _translation_deadline(self, sid):
        with self._translation_state_lock:
            queued_at = self._pending_final.get(sid, time.monotonic())
        return queued_at + self._translation_age_limit()

    def _run_translate_final(self, sid: str, text: str, context: list[str] | None = None) -> None:
        # The server has one inference slot. Finals own it ahead of optional previews.
        deadline = self._translation_deadline(sid)
        while not self._stop.is_set():
            if time.monotonic() >= deadline:
                self._translation_failed(sid, "翻译排队超过实时等待上限；原文已保留，可点补译")
                return
            if self._translate_gate.acquire(timeout=0.1):
                break
        else:
            return
        try:
            self._translate_final_locked(sid, text, context, deadline=deadline)
        finally:
            self._translate_gate.release()

    def _translate_final_locked(self, sid, text, context, *, deadline=None):
        if self._stop.is_set():
            return
        if self._uses_sakura():
            context = self._context_lines()
        translator = Translator(self.cfg)
        t0 = time.monotonic()
        if deadline is None:
            deadline = self._translation_deadline(sid)
        for attempt in range(2):
            parts = []
            last_emit = [0.0]
            def delta(piece):
                parts.append(piece)
                now = time.monotonic()
                if now - last_emit[0] >= 0.04:
                    self.sig_final_zh.emit(sid, ''.join(parts))
                    last_emit[0] = now
            control = TranslationStream(delta, self._stop.is_set,
                                        seconds=max(0.0, deadline - time.monotonic()),
                                        read_timeout=max(0.1, min(6.0, deadline - time.monotonic())))
            try:
                control.checkpoint()
                out = translator.translate(text, context or [], on_delta=control)
                control.checkpoint()
                break
            except TranslationCancelled:
                return
            except Exception as exc:
                if isinstance(exc, TranslationOutputError) and self._uses_sakura():
                    context = []
                # Retries share the original end-to-end budget, including queue
                # time. A pathological sentence can no longer block 2 x 90s.
                timed_out = isinstance(exc, TranslationDeadlineExceeded) or time.monotonic() >= deadline
                if (attempt or timed_out
                        or any(code in str(exc) for code in ("401", "403", "API Key", "额度"))):
                    message = ("翻译超过实时等待上限；原文已保留，可点补译"
                               if timed_out else str(exc))
                    self._translation_failed(sid, message, notify=not timed_out)
                    return
                if self._stop.wait(0.2):
                    return
        else:
            return
        if self._stop.is_set():
            return
        with self._translation_state_lock:
            self._failed_translations.pop(sid, None)
        self._last_final_zh.append(out)
        self._last_final_zh = self._last_final_zh[-8:]
        self.sig_final_zh.emit(sid, out)
        from .config import TRANSLATE_PRESETS
        label = TRANSLATE_PRESETS.get(translator.last_provider, {}).get("label", translator.last_provider)
        self.sig_translate_info.emit({"provider": translator.last_provider, "label": label,
                                      "seconds": round(time.monotonic() - t0, 1), "sid": sid,
                                      "engine": getattr(translator, "last_engine", ""),
                                      "engine_reason": getattr(translator, "last_engine_reason", "")})
        chosen = str(self.cfg.get("translate.provider", ""))
        if translator.last_provider and translator.last_provider != chosen and self._fallback_notified != chosen:
            self._fallback_notified = chosen
            self.sig_status.emit(f"翻译服务已使用配置允许的备用通道：{label}")

    def _publish_preview_source(self, gen: int, text: str) -> None:
        with self._preview_lock:
            if (self._stop.is_set() or self._pause.is_set() or self._reset_pending.is_set()
                    or not self._active or self._live['gen'] != gen or text == self._partial):
                return
            previous = self._partial
            self._partial = text
            self._preview_revision += 1
            # The UI already checks this id upon delivery. A queued delta
            # from the same utterance but an older ASR revision is now stale.
            sid = self._live['sid'] = f"L{gen}:{self._preview_revision}"
            source = text.strip()
            # Short initial guesses are noisy; a short *correction* to an
            # already requested source still needs to replace its old meaning.
            eligible = len(source) >= 6 or (source and self._live['text'])
            self._pending_preview = ((gen, sid, source, time.monotonic()) if eligible else None)
            self.sig_partial.emit(text)
            if previous and not text.startswith(previous):
                self.sig_live_zh.emit(sid, '')  # Not even a valid prefix any more.

    def _maybe_live_translate(self, now: float) -> None:
        if not self.cfg.get("translate.live", True) or not self.cfg.get("translate.enabled", True):
            with self._preview_lock:
                self._pending_preview = None
            return
        live = self._live
        interval = int(self.cfg.get("translate.live_min_interval_ms", 1100)) / 1000.0
        min_new = int(self.cfg.get("translate.live_min_new_chars", 2))
        live_provider = self._live_provider()
        main = str(self.cfg.get("translate.provider", ""))
        if live_provider == main:
            from .config import TRANSLATE_PRESETS

            kind = TRANSLATE_PRESETS.get(main, {}).get("kind")
            model = str(self.cfg.service(main).get('model') or '').lower()
            if kind == 'ollama' or (kind == 'llamacpp' and not any(name in model for name in ('hy-mt2', 'galtransl'))):
                interval = max(interval, 2.2)  # 较大的本地模型避免草稿堆积
            if TRANSLATE_PRESETS.get(main, {}).get("kind") == "localmt":
                interval = max(interval, 1.8)  # localmt 也是阻塞 CPU 的同步引擎，防止堆积
        if live_provider == main and self._has_final_pending():
            return
        with self._preview_lock:
            pending = self._pending_preview
            if pending is None or self._live_task_inflight:
                return
            gen, sid, text, changed = pending
            if (self._stop.is_set() or self._pause.is_set() or self._reset_pending.is_set()
                    or not self._active or gen != live['gen'] or sid != live['sid']):
                self._pending_preview = None
                return
            gap = max(interval, float(live.get("latency") or 0.0) * LIVE_DUTY_GAP)
            if now - live["last_req"] < gap:
                return
            previous = live['text']
            # Batch tiny appended suffixes, but translate even a one-character
            # change once stable. Rewrites and deletions are not suffixes.
            if (previous and text.startswith(previous) and 0 < len(text)-len(previous) < min_new
                    and now - changed < interval):
                return
            self._live_task_inflight = True
            self._pending_preview = None
            live.update(busy=True, last_req=now, text=text)
        if self._submit(self._live_pool, self._run_translate_live, gen, sid, text) is None:
            with self._preview_lock:
                self._live_task_inflight = False
                live["busy"] = False

    def _live_provider(self) -> str:
        return live_provider_for(self.cfg)

    def _run_translate_live(self, gen: int, sid: str, text: str) -> None:
        defer = False
        shared = False
        def obsolete():
            nonlocal defer
            if (self._stop.is_set() or self._pause.is_set() or not self._active
                    or gen != self._live["gen"] or sid != self._live["sid"]):
                return True
            if shared and self._has_final_pending():
                defer = True
                return True
            return False
        acquired = False
        ran = False
        t0 = time.monotonic()
        out = ""
        try:
            provider = self._live_provider()
            shared = provider == str(self.cfg.get("translate.provider", ""))
            if obsolete():
                return
            if shared:
                acquired = self._translate_gate.acquire(blocking=False)
                if not acquired:
                    defer = True
                    return
                if obsolete():
                    return
            translator = Translator(self.cfg)
            ran = True
            parts = []
            last_emit = [0.0]
            def delta(piece):
                parts.append(piece)
                now = time.monotonic()
                if now - last_emit[0] >= 0.04:
                    self.sig_live_zh.emit(sid, ''.join(parts))
                    last_emit[0] = now
            control = TranslationStream(delta, obsolete, seconds=10.0, read_timeout=6.0)
            out = translator.translate(text, self._context_lines(), provider=provider, on_delta=control)
            control.checkpoint()
            if out and out.count("*") < 4 and "＊＊" not in out:
                self.sig_live_zh.emit(sid, out)
                from .config import TRANSLATE_PRESETS
                label = TRANSLATE_PRESETS.get(translator.last_provider, {}).get("label", translator.last_provider)
                self.sig_translate_info.emit({"provider": translator.last_provider, "label": label,
                                              "seconds": round(time.monotonic() - t0, 1), "live": True,
                                              "engine": getattr(translator, "last_engine", ""),
                                              "engine_reason": getattr(translator, "last_engine_reason", "")})
        except TranslationCancelled:
            pass
        except Exception:
            # Optional preview failures must not stop final translation or occupy the slot forever.
            pass
        finally:
            if acquired:
                self._translate_gate.release()
            with self._preview_lock:
                self._live_task_inflight = False
                if gen == self._live["gen"]:
                    self._live["busy"] = False
                if (defer and self._pending_preview is None and sid == self._live["sid"]
                        and gen == self._live["gen"] and self._active
                        and not self._stop.is_set() and not self._pause.is_set()):
                    self._pending_preview = (gen, sid, text, t0)
                if ran:
                    spent = max(0.0, time.monotonic() - t0)
                    old = float(self._live.get("latency") or 0.0)
                    self._live["latency"] = spent if old <= 0 else round(old * 0.6 + spent * 0.4, 3)
