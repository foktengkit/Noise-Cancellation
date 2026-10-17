from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .denoise import AudioError, MAX_AUDIO_SECONDS, MAX_CHANNELS, MAX_SAMPLE_RATE
from .worker_security import PYTHON_FLAGS, worker_environment


PROCESS_SECONDS = 45
MAX_RESULT_BYTES = MAX_AUDIO_SECONDS * MAX_SAMPLE_RATE * MAX_CHANNELS * 2 + 128


def process_file(payload, filename, content_type, strength, hum_filter="off", engine="rnnoise"):
    with tempfile.TemporaryDirectory(prefix="noise-lift-") as directory:
        root = Path(directory)
        (root / "input.audio").write_bytes(payload)
        environment = worker_environment(root)
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        try:
            result = subprocess.run(
                [sys.executable, *PYTHON_FLAGS, "-m", "backend.worker", str(root), filename or "",
                 content_type or "", str(strength), hum_filter, engine], cwd=Path(__file__).resolve().parent.parent,
                env=environment, stdin=subprocess.DEVNULL, close_fds=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=PROCESS_SECONDS, **options)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Audio processing timed out.") from exc
        metadata_path = root / "result.json"
        if result.returncode != 0 or not metadata_path.is_file() or metadata_path.stat().st_size > 2048:
            raise AudioError("The audio file could not be processed safely.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "error" in metadata:
            raise AudioError(metadata["error"])
        outputs = []
        for name in ["original.wav", "denoised.wav"]:
            path = root / name
            if not path.is_file() or path.stat().st_size > MAX_RESULT_BYTES:
                raise AudioError("The decoded recording is too large.")
            outputs.append(path.read_bytes())
        return *outputs, metadata


def main(directory, filename, content_type, strength, hum_filter="off", engine="rnnoise"):
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_CPU, (PROCESS_SECONDS,) * 2)
    from .denoise import read_audio_bytes, reduce_noise, write_wav_bytes

    root = Path(directory)
    try:
        audio = read_audio_bytes((root / "input.audio").read_bytes(), filename, content_type)
        if engine not in ("rnnoise", "decode"):
            raise AudioError("Unsupported processing engine.")
        if engine == "decode" and len(audio.samples) > audio.sample_rate * 15:
            raise AudioError("Speaker separation accepts at most 15 seconds.")
        cleaned = reduce_noise(audio.samples, audio.sample_rate, float(strength), hum_filter)
        (root / "original.wav").write_bytes(write_wav_bytes(audio.samples, audio.sample_rate))
        (root / "denoised.wav").write_bytes(write_wav_bytes(cleaned, audio.sample_rate))
        metadata = {"sampleRate": audio.sample_rate, "channels": audio.channels,
                    "inputFrames": len(audio.samples),
                    "durationSeconds": round(audio.samples.shape[0] / audio.sample_rate, 2),
                    "strength": float(strength), "inputFormat": audio.format_name,
                    "engine": engine, "humFilter": hum_filter}
    except (AudioError, ValueError, MemoryError):
        metadata = {"error": "Please upload valid WAV, MP3, or M4A audio within the supported limits."}
    (root / "result.json").write_text(json.dumps(metadata), encoding="utf-8")


if __name__ == "__main__":
    os.umask(0o077)
    main(*sys.argv[1:])
