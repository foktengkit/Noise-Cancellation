"""Measure local synthetic noise cases without uploading or tuning the model."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import av
import numpy as np

from backend.denoise import StreamingNoiseReducer, read_audio_bytes, reduce_noise, write_wav_bytes
from backend.speech import MODEL_SHA256
from scripts.evaluate_denoising import si_sdr


NOISE_TYPES = ("white_fan", "hum", "wind_proxy", "clicks", "overlapping_speech_proxy")
STRENGTHS = (0.5, 0.85, 1.0)


def power(samples):
    return float(np.mean(np.asarray(samples, dtype=np.float64) ** 2))


def prepare_reference(samples, rate):
    speech = np.asarray(samples, dtype=np.float32)
    if speech.ndim == 2:
        speech = speech.mean(axis=1)
    if speech.ndim != 1 or not np.all(np.isfinite(speech)) or not len(speech):
        raise ValueError("A finite, nonempty clean speech reference is required.")
    peak = float(np.max(np.abs(speech)))
    if peak < 1e-6:
        raise ValueError("The clean reference is silent.")
    if len(speech) > rate * 118:
        raise ValueError("The clean reference must be at most 118 seconds before padding.")
    speech = np.pad(speech * (0.7 / peak), (rate, rate))
    window = np.ones(rate // 50, np.float32) / (rate // 50)
    active = np.convolve((np.abs(speech) > 0.015).astype(np.float32), window, "same") > 0.05
    if not np.any(active):
        raise ValueError("No active speech was found in the reference.")
    # Only padded silence is used, never a model-dependent voice activity mask.
    quiet = np.zeros(len(speech), dtype=bool)
    quiet[rate // 2:rate] = True
    quiet[-rate:-rate // 2] = True
    active[quiet] = False
    return speech, active, quiet


def build_noise(kind, reference, rate, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(len(reference)) / rate
    white = rng.normal(0, 1, len(reference))
    if kind == "white_fan":
        fan = sum(np.sin(2 * np.pi * frequency * t) / (index + 1)
                  for index, frequency in enumerate((70, 140, 210)))
        return white + 0.8 * fan
    if kind == "hum":
        return sum(np.sin(2 * np.pi * frequency * t) / (index + 1)
                   for index, frequency in enumerate((60, 120, 180))) + 0.05 * white
    if kind == "wind_proxy":
        frequencies = np.fft.rfftfreq(len(reference), 1 / rate)
        spectrum = np.fft.rfft(white) / np.sqrt(1 + (frequencies / 160) ** 4)
        low = np.fft.irfft(spectrum, n=len(reference))
        envelope = 0.3 + 0.7 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
        return low * envelope + 0.01 * white
    if kind == "clicks":
        noise = white * 0.02
        width = max(8, rate // 250)
        for start in rng.integers(0, len(noise) - width, max(2, int(len(noise) / rate * 4))):
            noise[start:start + width] += rng.normal(0, 4, width) * np.hanning(width)
        return noise
    if kind == "overlapping_speech_proxy":
        # A shifted copy is a stress proxy, not an independent-speaker dataset.
        return np.roll(reference, int(rate * rng.uniform(0.6, 1.8)))
    raise ValueError("Unknown synthetic noise type: " + kind)


def make_mixture(reference, noise, active, snr_db):
    reference, noise, active = np.asarray(reference), np.asarray(noise), np.asarray(active)
    if (reference.ndim != 1 or not len(reference) or noise.shape != reference.shape
            or active.shape != reference.shape or active.dtype != np.bool_
            or not np.any(active) or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(noise))):
        raise ValueError("Finite matching speech/noise arrays and a nonempty active mask are required.")
    with np.errstate(over="ignore", invalid="ignore"):
        speech_power, noise_power = power(reference[active]), power(noise[active])
    if not math.isfinite(speech_power) or not math.isfinite(noise_power) or min(speech_power, noise_power) <= 0:
        raise ValueError("Active speech and noise must have finite, positive energy.")
    try:
        snr_ratio = math.pow(10, float(snr_db) / 10)
    except (OverflowError, ValueError) as error:
        raise ValueError("SNR must be finite and within the numeric range.") from error
    denominator = snr_ratio * noise_power
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("SNR must be finite and within the numeric range.")
    gain = speech_power / denominator
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("The requested mixture gain is outside the numeric range.")
    noise = noise * np.sqrt(gain)
    mixture = reference + noise
    scale = min(1.0, 0.95 / max(float(np.max(np.abs(mixture))), 1e-15))
    return (reference * scale).astype(np.float32), (mixture * scale).astype(np.float32), scale


def measure(reference, estimate, active, quiet):
    estimate = np.asarray(estimate)
    if estimate.shape != reference.shape or not np.all(np.isfinite(estimate)):
        raise ValueError("Output length or finite-sample validation failed.")
    if float(np.max(np.abs(estimate))) > 1:
        raise ValueError("Output is outside the audio range.")
    silent = power(estimate) < 1e-15
    return {
        "siSdrDb": None if silent else round(si_sdr(reference, estimate), 3),
        "activeSiSdrDb": None if silent or power(estimate[active]) < 1e-15 else
        round(si_sdr(reference[active], estimate[active]), 3),
        "activeRmsGainDb": round(10 * np.log10(max(power(estimate[active]), 1e-15) /
                                               power(reference[active])), 3),
        "quietPower": power(estimate[quiet]),
        "silentOutput": silent,
        "peak": round(float(np.max(np.abs(estimate))), 6),
    }


def evaluate(path, output):
    payload = path.read_bytes()
    audio = read_audio_bytes(payload, path.name)
    rate = audio.sample_rate
    reference, active, quiet = prepare_reference(audio.samples, rate)
    output.mkdir(parents=True, exist_ok=True)
    (output / "reference.wav").write_bytes(write_wav_bytes(reference, rate))
    report = {
        "fixture": path.name, "fixtureSha256": hashlib.sha256(payload).hexdigest(),
        "modelSha256": MODEL_SHA256, "sampleRate": rate,
        "durationSeconds": round(len(reference) / rate, 3),
        "python": platform.python_version(), "pyav": av.__version__,
        "platform": platform.platform(), "cleanSpeech": [], "cases": [],
        "limitations": ["One local clean speech fixture, not a representative speech corpus.",
                        "Wind and overlapping speech are synthetic proxies, not real recordings.",
                        "SI-SDR is not perceived intelligibility; quiet attenuation is not speech-time noise attenuation.",
                        "Active RMS on mixtures includes remaining noise and is not isolated speech retention.",
                        "No subjective listening, ASR, PESQ or STOI evaluation."],
    }
    for level_db in (0, -12, -24):
        clean = reference * 10 ** (level_db / 20)
        (output / ("clean_level" + str(level_db) + "_original.wav")).write_bytes(write_wav_bytes(clean, rate))
        for strength in STRENGTHS:
            cleaned = reduce_noise(clean, rate, strength)[:, 0]
            report["cleanSpeech"].append({"inputLevelDb": level_db, "strength": strength,
                                          **measure(clean, cleaned, active, quiet)})
            if strength == 1:
                (output / ("clean_level" + str(level_db) + "_neural100.wav")).write_bytes(
                    write_wav_bytes(cleaned, rate))
    parity_input = None
    for kind in NOISE_TYPES:
        for seed in (420, 421):
            noise = build_noise(kind, reference, rate, seed)
            for snr in (-5, 5, 15):
                clean, noisy, scale = make_mixture(reference, noise, active, snr)
                case = {"noise": kind, "seed": seed, "activeInputSnrDb": snr,
                        "commonPeakScale": round(scale, 6), "input": measure(clean, noisy, active, quiet),
                        "outputs": []}
                save = seed == 420 and snr == 5
                if save:
                    (output / (kind + "_noisy.wav")).write_bytes(write_wav_bytes(noisy, rate))
                    (output / (kind + "_reference.wav")).write_bytes(write_wav_bytes(clean, rate))
                for strength in STRENGTHS:
                    start = time.perf_counter()
                    cleaned = reduce_noise(noisy, rate, strength)[:, 0]
                    elapsed = time.perf_counter() - start
                    metrics = measure(clean, cleaned, active, quiet)
                    delta = None if metrics["siSdrDb"] is None else metrics["siSdrDb"] - case["input"]["siSdrDb"]
                    quiet_power = case["input"]["quietPower"]
                    metrics.update({"strength": strength, "siSdrImprovementDb": None if delta is None else round(delta, 3),
                                    "quietAttenuationDb": None if quiet_power < 1e-15 else round(
                                        10 * np.log10(quiet_power / max(metrics["quietPower"], 1e-15)), 3),
                                    "processingSeconds": round(elapsed, 4)})
                    case["outputs"].append(metrics)
                    if save:
                        (output / (kind + "_neural" + str(round(strength * 100)) + ".wav")).write_bytes(
                            write_wav_bytes(cleaned, rate))
                report["cases"].append(case)
                if kind == "white_fan" and save:
                    parity_input = noisy
        print("Measured " + kind, flush=True)
    reducer = StreamingNoiseReducer(rate)
    chunks = [reducer.process(parity_input[i:i + rate // 4], 1)
              for i in range(0, len(parity_input), rate // 4)]
    streamed = np.concatenate([*chunks, reducer.finish()])
    file_output = reduce_noise(parity_input, rate, 1)[:, 0]
    np.testing.assert_allclose(streamed, file_output, atol=1e-6, rtol=0)
    zero = reduce_noise(parity_input, rate, 0)[:, 0]
    np.testing.assert_array_equal(zero, parity_input)
    report["streamParity"] = {"samples": len(streamed), "maxAbsoluteError": float(np.max(np.abs(streamed - file_output))),
                              "zeroStrengthExact": True}
    report["summary"] = []
    for kind in NOISE_TYPES:
        for strength in STRENGTHS:
            values = [row for case in report["cases"] if case["noise"] == kind
                      for row in case["outputs"] if row["strength"] == strength]
            deltas = [row["siSdrImprovementDb"] for row in values if row["siSdrImprovementDb"] is not None]
            pauses = [row["quietAttenuationDb"] for row in values if row["quietAttenuationDb"] is not None]
            report["summary"].append({"noise": kind, "strength": strength, "cases": len(values),
                                      "medianImprovementDb": round(float(np.median(deltas)), 3) if deltas else None,
                                      "minimumImprovementDb": min(deltas) if deltas else None,
                                      "worseCases": sum(value < 0 for value in deltas),
                                      "silentOutputs": sum(row["silentOutput"] for row in values),
                                      "medianQuietAttenuationDb": round(float(np.median(pauses)), 3) if pauses else None})
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cleanSpeech", "summary", "streamParity")}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clean", type=Path)
    parser.add_argument("--output", type=Path, default=Path("test-artifacts/quality-matrix"))
    args = parser.parse_args()
    evaluate(args.clean, args.output)
