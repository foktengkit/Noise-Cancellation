# Synthetic Speech Demo

The denoising examples contain a locally generated synthetic voice, not a person's
uploaded recording. It was produced with Windows System.Speech, Microsoft Hanhan
Desktop, at 16000 Hz mono. It is approximately 17.6 seconds long, including one
second of padding at each end.

The spoken text was authored for this project:

    Please check this voice recording. The fan is running in the background.
    We want to keep the words clear and reduce the noise. This is a test of
    speech enhancement, not active noise cancellation.

| File | Simulated background | Active-speech SNR | Seed |
| --- | --- | --- | --- |
| `sample-noisy.wav` | White noise and 70/140/210 Hz fan harmonics | 5 dB | 420 |
| `sample-hum.wav` | 60/120/180 Hz hum with faint white noise | 5 dB | 421 |
| `sample-wind.wav` | Modulated low-frequency noise, a wind proxy | 5 dB | 422 |
| `sample-keyboard.wav` | Short noise bursts, a keyboard-click proxy | 8 dB | 423 |
| `sample-strong.wav` | Strong white noise and fan harmonics | -3 dB | 424 |
| `sample-clean.wav` | None; clean speech control | N/A | N/A |

All files are approximately 17.6 seconds, 16 kHz mono PCM WAV. The same voice
reference is used for comparison; wind and keyboard sounds are simulations,
not field recordings. Mixtures are peak-scaled to avoid clipping. These demos
are not a representative quality benchmark or a promise of noise removal.

Regenerate from the repository root using an intentionally public synthetic
voice, never a private recording:

```powershell
powershell -NoProfile -File scripts/synthesize_test_speech.ps1 -Output test-artifacts/demo-clean.wav
.venv-secure/Scripts/python.exe -m scripts.create_sample test-artifacts/demo-clean.wav
```

The corresponding clean test fixture and evaluation outputs are local-only
under ignored `test-artifacts/`; only this intentionally public demo is bundled.
