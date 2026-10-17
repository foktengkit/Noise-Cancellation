"""Measure two explicit local references; keep source assignment permutation-aware."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from backend.denoise import read_audio_bytes, reduce_noise, write_wav_bytes
from scripts.evaluate_denoising import si_sdr
from separation_service.model import MODEL_RATE, REVISION, Separator
from separation_service.worker import resample_mono


def evaluate(directory, output, publish_sample=False):
    references = []
    for name in ("speaker-a.wav", "speaker-b.wav"):
        audio = read_audio_bytes((directory / name).read_bytes(), name)
        samples = resample_mono(audio.samples, audio.sample_rate)[:MODEL_RATE * 8]
        samples = np.pad(samples, (0, MODEL_RATE * 8 - len(samples)))
        references.append(samples * (0.35 / max(float(np.max(np.abs(samples))), 1e-8)))
    mixture = np.sum(references, axis=0).astype(np.float32)
    model = Separator(".tools/separation-model")
    start = time.perf_counter()
    separated = model.separate(mixture)
    elapsed = time.perf_counter() - start
    baseline = reduce_noise(mixture, MODEL_RATE, .85)[:, 0]
    assignments = [(0, 1), (1, 0)]
    scores = [[si_sdr(references[i], separated[:, assignment[i]]) for i in range(2)] for assignment in assignments]
    chosen = int(np.argmax([np.mean(values) for values in scores]))
    input_scores = [si_sdr(reference, mixture) for reference in references]
    report = {"modelRevision": REVISION, "durationSeconds": len(mixture) / MODEL_RATE,
              "torchVersion": importlib.metadata.version("torch"),
              "speechbrainVersion": importlib.metadata.version("speechbrain"),
              "sampleRate": MODEL_RATE, "processingSeconds": round(elapsed, 3),
              "inputSiSdrDb": input_scores, "rnnoise85SiSdrDb": [si_sdr(reference, baseline) for reference in references],
              "separatedSiSdrDb": scores[chosen], "referenceToSourceIndices": list(assignments[chosen]),
              "improvementDb": [scores[chosen][i] - input_scores[i] for i in range(2)],
              "limitations": "One two-voice TTS mixture; not real-speaker, Mandarin, reverb, or broad-domain validation."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "mixture.wav").write_bytes(write_wav_bytes(mixture, MODEL_RATE))
    (output / "rnnoise85.wav").write_bytes(write_wav_bytes(baseline, MODEL_RATE))
    for i in range(2):
        (output / ("reference-" + str(i + 1) + ".wav")).write_bytes(write_wav_bytes(references[i], MODEL_RATE))
        (output / ("source-" + str(i + 1) + ".wav")).write_bytes(write_wav_bytes(separated[:, i], MODEL_RATE))
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if publish_sample:
        Path("frontend/assets/sample-speakers.wav").write_bytes(write_wav_bytes(mixture, MODEL_RATE))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("test-artifacts/separation"))
    parser.add_argument("--output", type=Path, default=Path("test-artifacts/separation/results"))
    parser.add_argument("--publish-sample", action="store_true")
    args = parser.parse_args()
    evaluate(args.input, args.output, args.publish_sample)
