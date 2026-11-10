"""Play the supplied audio and measure real WASAPI capture clock drift locally."""
import os
import sys
from pathlib import Path
import json
import time
import threading
import wave

root = Path(__file__).resolve().parents[1]
installed = Path(os.environ['LOCALAPPDATA'])/'Programs/YishengTranslatorWhite/app'
sys.path[:0] = [str(installed), str(installed/'runtime/packages')]
import numpy as np
import soundcard as sc
from jpsub.core.audio import AudioCapture

out = root/'artifacts/loopback-probe'
out.mkdir(parents=True, exist_ok=True)
f = wave.open(str(root/'artifacts/replay.wav'), 'rb')
f.setpos(120*16000)
samples = np.frombuffer(f.readframes(300*16000), dtype='<i2').astype(np.float32)/32768
f.close()
frames = []
rows = []
started = time.monotonic()
total = 0
last = 0
def chunk(audio, rms):
    global total, last
    frames.append(audio.copy())
    total += len(audio)
    wall = time.monotonic()-started
    if wall-last >= 30:
        row = dict(wall=round(wall,3), captured_seconds=total/16000,
                   drift_seconds=round(wall-total/16000,3), rms=round(rms,5))
        rows.append(row)
        print(json.dumps(row),flush=True)
        last=wall

cap = AudioCapture(source='loopback', device_name='', on_chunk=chunk,
                   on_error=lambda text: print(text,flush=True), block_ms=80)
cap.start()
speaker = sc.default_speaker()
print('Testing default output: '+speaker.name, flush=True)
try:
    with speaker.player(samplerate=16000, channels=2, blocksize=1280) as player:
        for i in range(0,len(samples),1280):
            player.play(np.repeat(samples[i:i+1280,None],2,axis=1))
finally:
    cap.stop()
    cap.join(5)
    (out/'clock.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    if frames:
        captured=np.concatenate(frames)
        with wave.open(str(out/'captured.wav'),'wb') as target:
            target.setnchannels(1);target.setsampwidth(2);target.setframerate(16000)
            target.writeframes((np.clip(captured,-1,1)*32767).astype('<i2').tobytes())
