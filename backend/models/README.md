# Bundled Voice Model

`voice.rnnn` is the unmodified `sh.rnnn` model (somnolent-hogwash), trained for
speech against recording noise, from Gregor Richards' rnnoise-models repository.

- Upstream: https://github.com/GregorR/rnnoise-models
- Commit: `3eee541a283fd3b8f81b85b1748e3b9ccbefa04d`
- Source: https://raw.githubusercontent.com/GregorR/rnnoise-models/3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/somnolent-hogwash-2018-09-01/sh.rnnn
- Size: 297646 bytes
- SHA-256: `70bb6685eb0c2a1d18e2918dca3fbfbd39317010b1802eb1b6ea73a92f3fdec0`

The upstream author declares the model data noncreative and not subject to
copyright (excluding upstream tools/ and README). Those tools are not bundled.
See the author's declaration at:
https://github.com/GregorR/rnnoise-models/blob/3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/README.md

Inference uses FFmpeg's arnndn filter through the existing PyAV dependency.
No model code is executed, no models are downloaded at runtime, and no audio is
sent to an external AI service. This is a legacy RNNoise-compatible model, not
the newer RNNoise v0.2 architecture. Intermediate strength settings use a
speech-preserving power curve (`strength ** 1.75`) before the aligned dry/wet
mix; zero is unchanged audio and one is full neural output. Keep this notice
when redistributing.
