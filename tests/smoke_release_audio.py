"""Explicit offline release smoke: official file -> bundled ASR -> bundled CPU MT.

Run using the staged candidate's own python.exe -I -S -B, not the test runner.
No playback, capture, cloud provider or user history is involved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

TEST_WAV_SHA256 = "09abd330ce706a6e6969fe6bbc8275314af631fcee4e536afd0626570d263fbf"


def run(candidate: Path, wav: Path):
    candidate = candidate.resolve(strict=True)
    app = candidate / "app"
    packages = app / "runtime/packages"
    if Path(sys.executable).resolve() != (app / "runtime/py312/python.exe").resolve():
        raise RuntimeError("Use the exact candidate's bundled Python, not the installed or system Python.")
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise RuntimeError("Pass -I -S -B to keep other installations/user packages out.")
    if not (candidate / "manifest.json").is_file():
        raise RuntimeError("A completed candidate's release manifest is required.")
    with wav.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != TEST_WAV_SHA256:
            raise ValueError("Input is not the audited official Parakeet test_ja_1.wav.")

    def offline(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo"}:
            raise RuntimeError(f"Unexpected external operation in offline smoke: {event}")

    sys.addaudithook(offline)
    sys.path[:0] = [str(app), str(packages)]
    with tempfile.TemporaryDirectory(prefix=".audio-smoke-", dir=candidate.parent) as profile:
        os.environ["JPSUB_DATA_DIR"] = profile
        os.environ["JPSUB_MODEL_DIR"] = str(app / "models")
        from jpsub import paths
        from jpsub.core import asr, local_mt
        if (paths.USER_DATA != Path(profile) or paths.MODELS != app / "models" or
                asr.MODEL_DIR != paths.MODELS/"parakeet-ja-0.6b-int8" or
                local_mt.ARGOS_ROOT != paths.MODELS/"argos"):
            raise RuntimeError("Model or data directory escaped the candidate.")
        paths.ensure_dirs()
        samples = asr.decode_wav(wav)
        rec = asr.FinalASR(num_threads=4)
        start = time.monotonic()
        try:
            jp = rec.transcribe(samples).text
        finally:
            rec.close()
        asr_time = time.monotonic() - start
        if not re.search(r"[ぁ-ヿ]", jp):
            raise ValueError(f"No Japanese syllabary recognized: {jp!r}")
        start = time.monotonic()
        mt = local_mt.LocalMT("ja", "zh")
        zh = mt.translate(jp)
        mt_time = time.monotonic() - start
        if zh == jp or not re.search(r"[一-龥]", zh):
            raise ValueError(f"No distinct Chinese output: {zh!r}")
        for name in ("jpsub", "sherpa_onnx", "ctranslate2", "sentencepiece", "numpy", "soxr"):
            where = Path(sys.modules[name].__file__).resolve(strict=True)
            if not where.is_relative_to(app if name == "jpsub" else packages):
                raise RuntimeError(f"{name} loaded outside candidate: {where}")
        return dict(input_sha256=TEST_WAV_SHA256, duration_s=round(len(samples)/16000, 3),
                    asr_with_load_s=round(asr_time, 3), translation_with_load_s=round(mt_time, 3),
                    route=[f"{h['fr']}->{h['to']}" for h in mt._chain], jp=jp, zh=zh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--wav", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.candidate, args.wav), ensure_ascii=False, indent=2))
