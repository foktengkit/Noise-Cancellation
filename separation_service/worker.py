"""One isolated, time-bounded inference process per private separation request."""

from fractions import Fraction
import os
from pathlib import Path
import sys

import av
import numpy as np

from backend.denoise import read_wav_bytes, write_wav_bytes
from .model import MODEL_RATE, Separator


MAX_SECONDS = 15
PROCESS_SECONDS = 120


def resample_mono(samples, rate):
    mono = samples.mean(axis=1).astype(np.float32)
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=MODEL_RATE)
    frame = av.AudioFrame.from_ndarray(mono[None, :], format="fltp", layout="mono")
    frame.sample_rate, frame.time_base = rate, Fraction(1, rate)
    frames = [*resampler.resample(frame), *resampler.resample(None)]
    result = np.concatenate([item.to_ndarray()[0] for item in frames])
    return result[:round(len(mono) * MODEL_RATE / rate)].copy()


def main(directory, model_directory):
    if sys.platform != "win32":
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (PROCESS_SECONDS, PROCESS_SECONDS))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024,) * 2)
    root = Path(directory)
    audio = read_wav_bytes((root / "input.wav").read_bytes())
    if len(audio.samples) > audio.sample_rate * MAX_SECONDS:
        raise ValueError("The separation clip exceeds 15 seconds.")
    samples = resample_mono(audio.samples, audio.sample_rate)
    model = Separator(model_directory, os.environ.get("SEPARATION_DEVICE", "cpu"))
    result = model.separate(samples)
    (root / "sources.wav").write_bytes(write_wav_bytes(result, MODEL_RATE))


if __name__ == "__main__":
    os.umask(0o077)
    main(*sys.argv[1:])
