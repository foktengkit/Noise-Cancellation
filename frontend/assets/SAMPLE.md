# Synthetic Speech Demo

`sample-noisy.wav` contains a locally generated synthetic voice, not a person's
uploaded recording. It was produced with Windows System.Speech, Microsoft Hanhan
Desktop, at 16000 Hz mono. It is approximately 17.6 seconds long, including one
second of padding at each end.

The spoken text was authored for this project:

    Please check this voice recording. The fan is running in the background.
    We want to keep the words clear and reduce the noise. This is a test of
    speech enhancement, not active noise cancellation.

White noise and 70/140/210 Hz synthetic fan harmonics were added with NumPy
random seed 420 at 5 dB active-speech SNR. Generation and evaluation scripts:
`scripts/synthesize_test_speech.ps1` and `scripts/evaluate_denoising.py`.

The corresponding clean test fixture and evaluation outputs are local-only
under ignored `test-artifacts/`; only this intentionally public demo is bundled.
