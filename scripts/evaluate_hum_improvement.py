"""Compare the old full-strength preset with balanced, optional narrow notches."""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.denoise import read_audio_bytes, reduce_noise, write_wav_bytes
from scripts.evaluate_quality_matrix import make_mixture, measure, prepare_reference


def evaluate(path, output):
    audio = read_audio_bytes(path.read_bytes(), path.name)
    rate = audio.sample_rate
    speech, active, quiet = prepare_reference(audio.samples, rate)
    t = np.arange(len(speech)) / rate
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for frequency in (50, 60):
        for seed in (420, 421):
            rng = np.random.default_rng(seed)
            noise = sum(np.sin(2 * np.pi * frequency * harmonic * t) / harmonic for harmonic in (1, 2, 3))
            noise += .05 * rng.normal(0, 1, len(speech))
            for snr in (-5, 5, 15):
                clean, noisy, _ = make_mixture(speech, noise, active, snr)
                old = reduce_noise(noisy, rate, 1, "off")[:, 0]
                new = reduce_noise(noisy, rate, .85, str(frequency))[:, 0]
                old_metrics, new_metrics = [measure(clean, x, active, quiet) for x in (old, new)]
                old_attenuation, new_attenuation = [float(10 * np.log10(np.mean(noisy[quiet] ** 2) /
                                                                         max(np.mean(x[quiet] ** 2), 1e-15))) for x in (old, new)]
                rows.append({"frequency": frequency, "seed": seed, "inputSnrDb": snr,
                             "old": old_metrics, "new": new_metrics,
                             "oldQuietAttenuationDb": old_attenuation, "newQuietAttenuationDb": new_attenuation,
                             "siSdrChangeDb": new_metrics["siSdrDb"] - old_metrics["siSdrDb"]})
                if seed == 420 and snr == 5:
                    for name, samples in (("noisy", noisy), ("old100", old), ("new85", new), ("reference", clean)):
                        (output / (str(frequency) + "hz_" + name + ".wav")).write_bytes(write_wav_bytes(samples, rate))
    summary = [{"frequency": frequency,
                "oldMedianQuietAttenuationDb": float(np.median([r["oldQuietAttenuationDb"] for r in rows if r["frequency"] == frequency])),
                "newMedianQuietAttenuationDb": float(np.median([r["newQuietAttenuationDb"] for r in rows if r["frequency"] == frequency])),
                "medianSiSdrChangeDb": float(np.median([r["siSdrChangeDb"] for r in rows if r["frequency"] == frequency]))}
               for frequency in (50, 60)]
    clean = [{"humFilter": hum, **measure(speech, reduce_noise(speech, rate, .85, hum)[:, 0], active, quiet)}
             for hum in ("off", "50", "60")]
    report = {"fixture": path.name, "cases": rows, "summary": summary, "cleanSpeech": clean,
              "limitations": "One TTS reference and fixed synthetic hum; a manual mode, not automatic hum detection."}
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"summary": summary, "cleanSpeech": clean}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test-artifacts/hum-improvement"))
    args = parser.parse_args()
    evaluate(args.clean, args.output)
