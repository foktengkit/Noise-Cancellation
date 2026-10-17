"""Bounded, stateful RNNoise inference using the bundled FFmpeg filter."""

from collections import deque
from fractions import Fraction
from functools import lru_cache
import hashlib
from pathlib import Path

import av
import numpy as np

from .denoise import AudioError, _validate_audio_size, validate_hum_filter


MODEL_PATH = Path(__file__).resolve().parent / "models" / "voice.rnnn"
MODEL_SHA256 = "ae3f7411e1e6a884f839a4a145c394408398f09854dbc1216ee02faafc98a17b"


@lru_cache(maxsize=1)
def verify_model():
    if "arnndn" not in av.filter.filters_available:
        raise AudioError("This server does not support the speech denoising filter.")
    if not MODEL_PATH.is_file() or hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise AudioError("The bundled speech model is missing or damaged.")


class SpeechDenoiser:
    """One independent neural/filter state per recording, never shared by users."""

    def __init__(self, sample_rate, channels=1, hum_filter="off"):
        _validate_audio_size(1, sample_rate, channels)
        validate_hum_filter(hum_filter)
        verify_model()
        self.rate, self.channels = sample_rate, channels
        self.layout = "mono" if channels == 1 else "stereo"
        self.graph = av.filter.Graph()
        self.graph.threads = 1
        self.source = self.graph.add_abuffer(
            sample_rate=sample_rate, format="fltp", layout=self.layout,
            time_base=Fraction(1, sample_rate))
        up = self.graph.add("aresample", "48000")
        filters = [self.source, up]
        if hum_filter != "off":
            # Canonical frames keep IIR rounding independent of upload/WS chunk sizes.
            filters.append(self.graph.add("asetnsamples", n="480", p="0"))
            for harmonic in (1, 2, 3):
                filters.append(self.graph.add("bandreject", frequency=str(int(hum_filter) * harmonic),
                                               width_type="h", width="2", precision="f64"))
        neural = self.graph.add("arnndn", model=str(MODEL_PATH), mix="1")
        down = self.graph.add("aresample", str(sample_rate))
        planar = self.graph.add("aformat", sample_fmts="fltp")
        self.sink = self.graph.add("abuffersink")
        self.graph.link_nodes(*filters, neural, down, planar, self.sink)
        self.graph.configure()
        self.pending = deque()
        self.pending_offset = 0
        self.received = self.emitted = self.pushed = 0
        # arnndn emits audio one 480-sample / 48 kHz frame behind its input.
        self.skip = round(sample_rate / 100)
        self.finished = False

    def process(self, samples, strength):
        if self.finished:
            raise AudioError("This recording has already ended.")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise AudioError("Invalid audio channel shape.")
        if not 0 < len(samples) <= self.rate * 2:
            raise AudioError("Speech inference accepts at most two seconds per block.")
        strength = float(strength)
        if not np.isfinite(strength) or not 0 <= strength <= 1 or not np.all(np.isfinite(samples)):
            raise AudioError("Audio samples and strength must be finite and within range.")
        _validate_audio_size(self.received + len(samples), self.rate, self.channels)
        self.received += len(samples)
        self.pending.append((samples.copy(), strength))
        self._push(samples)
        return self._pull()

    def finish(self):
        if self.finished:
            return np.empty((0, self.channels), dtype=np.float32)
        self.finished = True
        # Padding drains look-ahead while avoiding the partial-EOF path that
        # caused a native crash in the tested FFmpeg build.
        self._push(np.zeros((self.rate // 10, self.channels), dtype=np.float32))
        output = self._pull()
        if self.emitted != self.received:
            raise AudioError("The speech filter returned incomplete audio.")
        return output

    def _push(self, samples):
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(samples.T), format="fltp", layout=self.layout)
        frame.sample_rate = self.rate
        frame.time_base = Fraction(1, self.rate)
        frame.pts = self.pushed
        self.pushed += len(samples)
        self.source.push(frame)

    def _pull(self):
        chunks = []
        while True:
            try:
                frame = self.sink.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                break
            wet = np.ascontiguousarray(frame.to_ndarray().T, dtype=np.float32)
            if wet.ndim != 2 or wet.shape[1] != self.channels or not np.all(np.isfinite(wet)):
                raise AudioError("The speech filter returned invalid audio samples.")
            skip = min(self.skip, len(wet))
            wet, self.skip = wet[skip:], self.skip - skip
            position = 0
            while position < len(wet) and self.pending:
                dry, strength = self.pending[0]
                count = min(len(wet) - position, len(dry) - self.pending_offset)
                original = dry[self.pending_offset:self.pending_offset + count]
                # Blend aligned samples, not loudness: zero strength is bit-exact.
                mixed = original if strength == 0 else (
                    original * (1 - strength) + wet[position:position + count] * strength)
                chunks.append(mixed.copy() if strength == 0 else np.clip(mixed, -1, 1).astype(np.float32))
                position += count
                self.emitted += count
                self.pending_offset += count
                if self.pending_offset == len(dry):
                    self.pending.popleft()
                    self.pending_offset = 0
        return np.concatenate(chunks) if chunks else np.empty((0, self.channels), dtype=np.float32)
