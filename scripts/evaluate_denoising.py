"""Evaluate a local clean speech fixture; never sends audio to external services."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from backend.denoise import read_audio_bytes, reduce_noise, reduce_stationary_noise, write_wav_bytes


def si_sdr(reference, estimate):
    reference = reference.astype(np.float64) - np.mean(reference)
    estimate = estimate.astype(np.float64) - np.mean(estimate)
    target = reference * (np.sum(estimate * reference) / max(np.sum(reference ** 2), 1e-15))
    return float(10 * np.log10(max(np.sum(target ** 2), 1e-15) /
                             max(np.sum((estimate - target) ** 2), 1e-15)))


def evaluate(path, output):
    audio = read_audio_bytes(path.read_bytes(), path.name)
    rate = audio.sample_rate
    speech = audio.samples.mean(axis=1)
    speech = speech * (0.7 / max(np.max(np.abs(speech)), 1e-8))
    speech = np.pad(speech, (rate, rate))
    active = np.convolve((np.abs(speech) > 0.015).astype(np.float32),
                         np.ones(rate // 50, np.float32) / (rate // 50), "same") > 0.05
    rng = np.random.default_rng(420)
    t = np.arange(len(speech)) / rate
    fan = sum(np.sin(2 * np.pi * frequency * t) / (index + 1)
              for index, frequency in enumerate((70, 140, 210)))
    noise = rng.normal(0, 1, len(speech)) + 0.8 * fan
    noise *= np.sqrt(np.mean(speech[active] ** 2) / (10 ** (5 / 10) * np.mean(noise[active] ** 2)))
    noisy = (speech + noise).astype(np.float32)
    if np.max(np.abs(noisy)) >= 1:
        raise ValueError("Fixture clips. Reduce speech/noise levels.")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    cleaned = reduce_noise(noisy, rate, 1)[:, 0]
    elapsed = time.perf_counter() - start
    old = reduce_stationary_noise(noisy, rate, .82)[:, 0]
    clean_only = reduce_noise(speech.astype(np.float32), rate, 1)[:, 0]
    quiet = np.r_[rate // 2:rate, len(speech) - rate:len(speech) - rate // 2]
    report = {
        "fixture": path.name, "sampleRate": rate, "durationSeconds": round(len(speech) / rate, 3),
        "noise": "seed=420 white noise + 70/140/210 Hz fan; active-speech SNR 5 dB",
        "inputSiSdrDb": round(si_sdr(speech, noisy), 2),
        "oldSpectralSiSdrDb": round(si_sdr(speech, old), 2),
        "neuralSiSdrDb": round(si_sdr(speech, cleaned), 2),
        "quietNoiseAttenuationDb": round(float(10 * np.log10(np.mean(noisy[quiet] ** 2) /
                                                               max(np.mean(cleaned[quiet] ** 2), 1e-15))), 2),
        "oldQuietNoiseAttenuationDb": round(float(10 * np.log10(np.mean(noisy[quiet] ** 2) /
                                                                  max(np.mean(old[quiet] ** 2), 1e-15))), 2),
        "cleanSpeechSiSdrDb": round(si_sdr(speech, clean_only), 2),
        "cleanSpeechGainDb": round(float(20 * np.log10(max(np.sqrt(np.mean(clean_only[active] ** 2)), 1e-15) /
                                                        np.sqrt(np.mean(speech[active] ** 2)))), 2),
        "processingSeconds": round(elapsed, 3),
    }
    for name, samples in (("reference", speech), ("noisy", noisy), ("spectral", old), ("neural", cleaned)):
        (output / (name + ".wav")).write_bytes(write_wav_bytes(samples, rate))
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["neuralSiSdrDb"] < report["inputSiSdrDb"] + 2:
        raise AssertionError("Speech quality improvement is below the 2 dB acceptance margin.")
    if report["cleanSpeechGainDb"] < -6:
        raise AssertionError("Too much clean speech was removed.")
    if report["quietNoiseAttenuationDb"] < 12:
        raise AssertionError("Background noise suppression is below the 12 dB acceptance margin.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test-artifacts/quality"))
    args = parser.parse_args()
    evaluate(args.clean, args.output)
