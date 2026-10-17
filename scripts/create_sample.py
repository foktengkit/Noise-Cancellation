"""Generate a deterministic synthetic melody with background noise for the UI demo."""

from pathlib import Path

import numpy as np

from backend.denoise import write_wav_bytes


def main():
    rate = 16000
    time = np.arange(rate * 8, dtype=np.float32) / rate
    rng = np.random.default_rng(42)
    samples = 0.045 * rng.standard_normal(time.size) + 0.025 * np.sin(2 * np.pi * 60 * time)
    for index, frequency in enumerate([196, 246.94, 293.66, 329.63, 293.66, 246.94]):
        start = 0.8 + index * 1.05
        local_time = time - start
        active = (local_time >= 0) & (local_time < 0.7)
        note_time = local_time[active]
        envelope = np.minimum(note_time / 0.03, 1) * np.minimum((0.7 - note_time) / 0.16, 1)
        note = np.sin(2 * np.pi * frequency * note_time) + 0.3 * np.sin(2 * np.pi * frequency * 2 * note_time)
        samples[active] += 0.28 * envelope * note
    output = Path(__file__).resolve().parents[1] / "frontend" / "assets" / "sample-noisy.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(write_wav_bytes(samples.astype(np.float32), rate))
    print("Generated frontend/assets/sample-noisy.wav (8 seconds, synthetic audio).")


if __name__ == "__main__":
    main()
