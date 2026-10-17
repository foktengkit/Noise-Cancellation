"""Build public demo variations from an explicitly supplied synthetic voice."""

import argparse
from pathlib import Path

import numpy as np

from backend.denoise import read_audio_bytes, write_wav_bytes
from scripts.evaluate_quality_matrix import build_noise, make_mixture, prepare_reference


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("sample-noisy.wav", "white_fan", 5, 420),
    ("sample-hum.wav", "hum", 5, 421),
    ("sample-wind.wav", "wind_proxy", 5, 422),
    ("sample-keyboard.wav", "clicks", 8, 423),
    ("sample-strong.wav", "white_fan", -3, 424),
    ("sample-clean.wav", None, None, None),
)


def create_samples(clean, output):
    audio = read_audio_bytes(clean.read_bytes(), clean.name)
    if audio.sample_rate != 16000 or audio.channels != 1:
        raise ValueError("Use a 16 kHz mono synthetic speech reference.")
    reference, active, _ = prepare_reference(audio.samples, audio.sample_rate)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, kind, snr, seed in CASES:
        if kind is None:
            mixture = reference.astype(np.float32)
        else:
            noise = build_noise(kind, reference, audio.sample_rate, seed)
            _, mixture, _ = make_mixture(reference, noise, active, snr)
        path = output / filename
        path.write_bytes(write_wav_bytes(mixture, audio.sample_rate))
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clean", type=Path, help="Synthetic speech only; never supply a private recording.")
    parser.add_argument("--output", type=Path, default=ROOT / "frontend/assets")
    args = parser.parse_args()
    for path in create_samples(args.clean, args.output):
        print("Generated " + str(path))
