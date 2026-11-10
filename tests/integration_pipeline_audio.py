"""Offline integration: real WAV -> VAD -> Parakeet -> mocked translation.

Run with the bundled Python and application/package directories on PYTHONPATH.
No audio device, GPU, internet or personal configuration is accessed.
"""
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PySide6 import QtCore

from jpsub.core import pipeline
from jpsub.core.asr import decode_wav
from jpsub.paths import MODELS


class TestConfig:
    def get(self, key, default=None):
        return {
            'asr.partial': False,
            'asr.final_threads': 6,
            'asr.max_segment_s': 10,
            'translate.enabled': True,
            'translate.live': False,
            'translate.provider': 'custom',
        }.get(key, default)

    def service(self, name):
        return {}


class NoDeviceCapture:
    def __init__(self, **kwargs):
        pass
    def start(self):
        pass
    def stop(self):
        pass
    def join(self, timeout=None):
        pass


class OfflineTranslator:
    last_provider = 'custom'
    last_engine = ''
    last_engine_reason = ''
    def __init__(self, cfg):
        pass
    def translate(self, text, context, on_delta=None, **kwargs):
        if on_delta:
            on_delta('测试译文：')
            on_delta(text)
        return '测试译文：' + text


def main():
    app = QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])
    job = pipeline.Pipeline(TestConfig())
    segments, translations, errors, infos = [], {}, [], []
    job.sig_segment.connect(segments.append, QtCore.Qt.DirectConnection)
    job.sig_error.connect(errors.append, QtCore.Qt.DirectConnection)
    job.sig_final_zh.connect(lambda sid, text: translations.update({sid: text}), QtCore.Qt.DirectConnection)
    job.sig_translate_info.connect(infos.append, QtCore.Qt.DirectConnection)
    started = time.perf_counter()
    duration = 0.0
    with patch.object(pipeline, 'AudioCapture', NoDeviceCapture), patch.object(pipeline, 'Translator', OfflineTranslator):
        try:
            job._setup()
            for path in sorted((MODELS / 'parakeet-ja-0.6b-int8/test_wavs').glob('*.wav')):
                audio = decode_wav(path)
                duration += len(audio) / 16000.0
                before = len(segments)
                with_silence = np.concatenate([audio, np.zeros(32000, dtype=np.float32)])
                for offset in range(0, len(with_silence), 1280):
                    chunk = with_silence[offset:offset+1280]
                    job._process_chunk(chunk, float(np.sqrt(np.mean(chunk ** 2))))
                job._final_pool.submit(lambda: None).result(timeout=120)
                job._trans_pool.submit(lambda: None).result(timeout=20)
                assert len(segments) > before, f'No speech recognized from {path.name}'
            assert not errors, errors
            assert len(translations) == len(segments), (len(segments), len(translations))
            assert all(translations[s.sid] == '测试译文：' + s.jp for s in segments)
            assert len(infos) == len(segments), (len(infos), len(segments))
            assert not job._pending_final
            assert not job._failed_translations
            assert all(s.t1 > s.t0 >= 0 for s in segments)
        finally:
            job._stop.set()
            job._shutdown_pools()
            job._release_resources()
    assert job._final is None and job._capture is None and job._vad is None
    assert job._queue.empty() and job._pools_closed
    result = {
        'audio_seconds': duration,
        'elapsed_seconds': time.perf_counter() - started,
        'segments': [{'jp': s.jp, 't0': s.t0, 't1': s.t1, 'asr_seconds': s.asr_seconds} for s in segments],
        'completed_translations': len(infos),
        'errors': errors,
        'released_resources': True,
        'translation_mocked': True,
    }
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
