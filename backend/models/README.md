# Bundled Voice Model

`voice.rnnn` is the unmodified `bd.rnnn` model (beguiling-drafter), trained for
voice against recording noise, from Gregor Richards' rnnoise-models repository.

- Upstream: https://github.com/GregorR/rnnoise-models
- Commit: `3eee541a283fd3b8f81b85b1748e3b9ccbefa04d`
- Source: https://raw.githubusercontent.com/GregorR/rnnoise-models/3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/beguiling-drafter-2018-08-30/bd.rnnn
- Size: 299693 bytes
- SHA-256: `ae3f7411e1e6a884f839a4a145c394408398f09854dbc1216ee02faafc98a17b`

The upstream author declares the model data noncreative and not subject to
copyright (excluding upstream tools/ and README). Those tools are not bundled.
See the author's declaration at:
https://github.com/GregorR/rnnoise-models/blob/3eee541a283fd3b8f81b85b1748e3b9ccbefa04d/README.md

Inference uses FFmpeg's arnndn filter through the existing PyAV dependency.
No model code is executed, no models are downloaded at runtime, and no audio is
sent to an external AI service. This is a legacy RNNoise-compatible model, not
the newer RNNoise v0.2 architecture. Keep this notice when redistributing.
