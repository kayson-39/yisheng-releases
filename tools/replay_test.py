"""Local real-time video replay through production VAD/ASR/translation workers.

Only replaces the sound-device capture with paced 16 kHz PCM; models, settings,
draft competition and translation scheduling are unchanged. No network upload.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import wave

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--installed', required=True)
p.add_argument('--wav', required=True)
p.add_argument('--out', required=True)
p.add_argument('--seconds', type=float, default=600)
p.add_argument('--offset', type=float, default=0)
args = p.parse_args()
installed = Path(args.installed)
out = Path(args.out).resolve()
out.mkdir(parents=True, exist_ok=True)
os.environ['JPSUB_DATA_DIR'] = str(out / 'userdata')
os.environ['JPSUB_MODEL_DIR'] = str(installed / 'models')
sys.dont_write_bytecode = True
sys.path[:0] = [str(Path(args.source).resolve()), str(installed / 'runtime/packages')]
import numpy as np
import psutil
from PySide6 import QtCore
from jpsub.core import pipeline, llama_engine
from jpsub.core.config import Config

llama_engine.ENGINE_DIR = installed / 'engine'
llama_engine.ROOT = installed
cfg = Config(Path(os.environ['LOCALAPPDATA']) / 'YishengTranslator/UserData/config.json')
if cfg.get('translate.provider') != 'llamacpp':
    raise RuntimeError('Replay requires the configured local translation provider')
cfg.set('translate.auto_fallback', False, save=False)
cfg.set('translate.live_provider', 'same', save=False)
app = QtCore.QCoreApplication([])
capture_started = [None]
feed_done = threading.Event()
lock = threading.RLock()
events = []
segments = {}

def record(kind, **data):
    with lock:
        row = dict(kind=kind, wall=round(time.monotonic() - start, 3), **data)
        events.append(row)
        log.write(json.dumps(row, ensure_ascii=False) + '\n')
        log.flush()

class ReplayCapture(threading.Thread):
    def __init__(self, on_chunk, **kw):
        super().__init__(daemon=True)
        self.on_chunk = on_chunk
        self.stopped = threading.Event()
    def stop(self):
        self.stopped.set()
    def run(self):
        capture_started[0] = time.monotonic()
        with wave.open(args.wav, 'rb') as f:
            assert f.getframerate() == 16000 and f.getnchannels() == 1
            f.setpos(min(f.getnframes(), int(args.offset * 16000)))
            fed = 0
            while fed < args.seconds * 16000 and not self.stopped.is_set():
                data = f.readframes(1280)
                if not data:
                    break
                audio = np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768
                self.on_chunk(audio, float(np.sqrt(np.mean(audio ** 2))))
                fed += len(audio)
                self.stopped.wait(max(0, capture_started[0] + fed / 16000 - time.monotonic()))
        record('feed_done', audio_seconds=fed / 16000)
        feed_done.set()

pipeline.AudioCapture = ReplayCapture
job = pipeline.Pipeline(cfg)
start = time.monotonic()
log = (out / 'events.jsonl').open('w', encoding='utf-8')
job.sig_status.connect(lambda text: record('status', text=text), QtCore.Qt.DirectConnection)
job.sig_error.connect(lambda text: record('error', text=text), QtCore.Qt.DirectConnection)
job.sig_translate_queue.connect(lambda data: record('queue', **data), QtCore.Qt.DirectConnection)
job.sig_translate_failed.connect(lambda sid, text: record('failed', sid=sid, text=text), QtCore.Qt.DirectConnection)
def segment(seg):
    segments[seg.sid] = dict(t0=seg.t0, t1=seg.t1, jp=seg.jp)
    record('segment', sid=seg.sid, t0=seg.t0, t1=seg.t1, jp=seg.jp,
           lag=time.monotonic() - capture_started[0] - seg.t1)
job.sig_segment.connect(segment, QtCore.Qt.DirectConnection)
def translated(data):
    sid = data.get('sid')
    if sid in segments:
        data = dict(data, lag=round(time.monotonic() - capture_started[0] - segments[sid]['t1'], 3))
    record('translated', **data)
job.sig_translate_info.connect(translated, QtCore.Qt.DirectConnection)
job.sig_final_zh.connect(lambda sid, text: record('text', sid=sid, text=text), QtCore.Qt.DirectConnection)
job.sig_stats.connect(lambda data: record('asr', **data), QtCore.Qt.DirectConnection)
job.start()
process = psutil.Process()
next_sample = 0
drain_start = None
try:
    while job.isRunning():
        app.processEvents()
        elapsed = time.monotonic() - start
        if elapsed >= next_sample:
            children = process.children(recursive=True)
            mem = process.memory_info().rss
            for child in children:
                try: mem += child.memory_info().rss
                except psutil.Error: pass
            record('resource', rss_mb=round(mem/1048576, 1), audio_queue=job._queue.qsize(),
                   final_pending=len(job._pending_final), threads=process.num_threads())
            print(json.dumps(events[-1], ensure_ascii=False), flush=True)
            next_sample += 30
        if feed_done.is_set():
            drain_start = drain_start or time.monotonic()
            if time.monotonic()-drain_start > 20:
                job.stop(0)
        if elapsed > args.seconds + 180:
            job.stop(0)
        time.sleep(0.02)
finally:
    job.stop(0)
    job.wait(120000)
    llama_engine.stop_all()
    finals = [e for e in events if e['kind']=='translated' and 'sid' in e]
    lags = [e['lag'] for e in finals if 'lag' in e]
    queues = [e for e in events if e['kind']=='queue']
    summary = dict(source=args.source, seconds=args.seconds, offset=args.offset,
                   segment_count=len(segments), completed=len(finals),
                   max_pending=max((e['pending'] for e in queues), default=0),
                   max_queue_age=max((e['oldest_s'] for e in queues), default=0),
                   lag_p50=float(np.percentile(lags,50)) if lags else None,
                   lag_p95=float(np.percentile(lags,95)) if lags else None,
                   lag_max=max(lags, default=None),
                   failures=[e for e in events if e['kind'] in ('error','failed')])
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    log.close()
