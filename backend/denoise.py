from __future__ import annotations

import io
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import av
    from av.error import FFmpegError
except ImportError:  # pragma: no cover - handled at runtime with a clear API error.
    av = None

    class FFmpegError(Exception):
        pass


class AudioError(ValueError):
    """Raised when the uploaded audio cannot be processed."""


SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a"}
MAX_AUDIO_SECONDS = 120
MAX_SAMPLE_RATE = 48000
MAX_CHANNELS = 2
DEFAULT_STRENGTH = 0.85
HUM_FILTERS = ("off", "50", "60")


def validate_hum_filter(value):
    if not isinstance(value, str) or value not in HUM_FILTERS:
        raise AudioError("Hum filter must be off, 50, or 60.")
    return value


@dataclass(frozen=True)
class AudioData:
    samples: np.ndarray
    sample_rate: int
    channels: int
    format_name: str = "wav"


def read_audio_bytes(data: bytes, filename: str | None = None, content_type: str | None = None) -> AudioData:
    """Read WAV, MP3, or M4A bytes into float32 samples shaped as (frames, channels)."""
    extension = _extension_from_filename(filename)
    if extension and extension not in SUPPORTED_EXTENSIONS:
        raise AudioError("Please upload a WAV, MP3, or M4A audio file.")

    if extension == ".wav" or _looks_like_wav(data):
        return read_wav_bytes(data)

    return read_media_bytes(data, filename=filename, content_type=content_type)


def read_wav_bytes(data: bytes) -> AudioData:
    """Read PCM WAV bytes into float32 samples shaped as (frames, channels)."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            frames = wav.getnframes()
            _validate_audio_size(frames, sample_rate, channels)
            pcm = wav.readframes(frames)
    except (wave.Error, EOFError) as exc:
        raise AudioError("Please upload a valid PCM WAV file.") from exc

    if channels < 1:
        raise AudioError("The WAV file does not contain any audio channels.")
    if sample_rate <= 0:
        raise AudioError("The WAV file has an invalid sample rate.")

    if len(pcm) != frames * channels * sample_width:
        raise AudioError("The WAV data is truncated.")
    samples = _pcm_to_float32(pcm, sample_width)
    if samples.size % channels != 0:
        raise AudioError("The WAV data is malformed.")

    return AudioData(samples=samples.reshape(-1, channels), sample_rate=sample_rate, channels=channels)


def read_media_bytes(data: bytes, filename: str | None = None, content_type: str | None = None) -> AudioData:
    """Decode compressed audio formats through PyAV/FFmpeg."""
    if av is None:
        raise AudioError("MP3/M4A support requires PyAV. Run pip install -r requirements.txt.")

    try:
        extension = _extension_from_filename(filename)
        if extension not in {".mp3", ".m4a"}:
            raise AudioError("A WAV, MP3, or M4A filename is required.")
        if extension == ".m4a":
            _validate_m4a_boxes(data)
        with av.open(io.BytesIO(data), mode="r", format="mp3" if extension == ".mp3" else "mov",
                     options={"protocol_whitelist": "", "format_whitelist": "mp3,mov",
                              "probesize": "1048576", "analyzeduration": "5000000"}) as container:
            stream = next((item for item in container.streams if item.type == "audio"), None)
            if stream is None:
                raise AudioError("The uploaded file does not contain an audio stream.")

            sample_rate = int(stream.codec_context.sample_rate or 0)
            resampler = None
            chunks: list[np.ndarray] = []
            total_frames = 0
            largest_frame = 0

            for frame in _decode_frames(container, stream):
                if sample_rate <= 0:
                    sample_rate = int(frame.sample_rate or 0)
                if sample_rate <= 0:
                    raise AudioError("The audio file has an invalid sample rate.")
                largest_frame = max(largest_frame, frame.samples)

                if resampler is None:
                    layout = frame.layout.name if frame.layout else "stereo"
                    _validate_audio_size(1, sample_rate, len(frame.layout.channels))
                    resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)

                for resampled in resampler.resample(frame):
                    total_frames += resampled.samples
                    _validate_audio_size(total_frames, sample_rate, len(resampled.layout.channels))
                    chunks.append(_float_planar_frame_to_samples(resampled))

            if resampler is not None:
                for resampled in resampler.resample(None):
                    total_frames += resampled.samples
                    _validate_audio_size(total_frames, sample_rate, len(resampled.layout.channels))
                    chunks.append(_float_planar_frame_to_samples(resampled))
            if extension == ".m4a" and stream.duration is not None and stream.duration > 0 and stream.time_base:
                expected_frames = round(stream.duration * stream.time_base * sample_rate)
                # AAC priming/padding and resampling can differ by a codec frame.
                tolerance = max(1, largest_frame, stream.codec_context.frame_size) * 2
                if total_frames + tolerance < expected_frames:
                    raise AudioError("The M4A audio data is incomplete or truncated.")
    except AudioError:
        raise
    except FFmpegError as exc:
        raise AudioError("The audio file could not be decoded. Please try a WAV, MP3, or M4A file.") from exc

    if not chunks:
        raise AudioError("The uploaded file does not contain decodable audio.")

    samples = np.concatenate(chunks, axis=0)
    if samples.size == 0:
        raise AudioError("The uploaded file does not contain decodable audio.")

    return AudioData(
        samples=samples,
        sample_rate=sample_rate,
        channels=samples.shape[1],
        format_name=_format_name(filename, content_type),
    )


def write_wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Write float samples shaped as (frames, channels) to 16-bit PCM WAV bytes."""
    if samples.ndim == 1:
        samples = samples[:, None]

    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(samples.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        write_wav_frames(wav, samples)
    return output.getvalue()


def write_wav_frames(output, samples: np.ndarray) -> None:
    """Quantize in bounded blocks instead of copying a whole recording."""
    for start in range(0, len(samples), 16384):
        block = samples[start:start + 16384]
        if not np.all(np.isfinite(block)):
            raise AudioError("Audio samples must be finite before encoding.")
        block = np.clip(block, -1.0, 1.0)
        output.writeframesraw((block * 32767.0).astype("<i2").tobytes())


def _validate_m4a_boxes(data: bytes) -> None:
    """Check outer box bounds; FFmpeg remains responsible for media parsing."""
    offset = 0
    while offset < len(data):
        if len(data) - offset < 8:
            raise AudioError("The M4A container is incomplete or malformed.")
        size = struct.unpack_from(">I", data, offset)[0]
        header_size = 8
        if size == 1:
            header_size = 16
            if len(data) - offset < header_size:
                raise AudioError("The M4A container is incomplete or malformed.")
            size = struct.unpack_from(">Q", data, offset + 8)[0]
        elif size == 0:
            return
        if size < header_size or size > len(data) - offset:
            raise AudioError("The M4A container is incomplete or malformed.")
        offset += size


def _decode_frames(container, stream):
    for packet in container.demux(stream):
        if packet.is_corrupt:
            raise AudioError("The audio file contains corrupt packets.")
        yield from packet.decode()


def reduce_noise(samples: np.ndarray, sample_rate: int, strength: float = DEFAULT_STRENGTH,
                 hum_filter: str = "off") -> np.ndarray:
    """Suppress background noise while retaining speech; preserve rate and channels."""
    from .speech import SpeechDenoiser

    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise AudioError("Invalid audio channel shape.")
    if not np.isfinite(strength) or not 0 <= strength <= 1 or not np.all(np.isfinite(samples)):
        raise AudioError("Audio samples and strength must be finite and within range.")
    _validate_audio_size(len(samples), sample_rate, samples.shape[1])
    validate_hum_filter(hum_filter)
    if strength == 0:
        return samples.copy()
    reducer = SpeechDenoiser(sample_rate, samples.shape[1], hum_filter)
    output = np.empty_like(samples)
    written = 0
    for start in range(0, len(samples), sample_rate * 2):
        cleaned = reducer.process(samples[start:start + sample_rate * 2], strength)
        output[written:written + len(cleaned)] = cleaned
        written += len(cleaned)
    tail = reducer.finish()
    output[written:written + len(tail)] = tail
    if written + len(tail) != len(samples):
        raise AudioError("The speech filter returned incomplete audio.")
    return output


def reduce_stationary_noise(samples: np.ndarray, sample_rate: int, strength: float = 0.82) -> np.ndarray:
    """
    Reduce stationary background noise with a lightweight spectral gate.

    strength ranges from 0.0 to 1.0. Higher values remove more noise but may
    create a more processed sound.
    """
    if samples.ndim == 1:
        samples = samples[:, None]

    strength = float(np.clip(strength, 0.0, 1.0))
    if not np.isfinite(strength) or not np.all(np.isfinite(samples)):
        raise AudioError("Audio samples and strength must be finite.")
    _validate_audio_size(samples.shape[0], sample_rate, samples.shape[1])
    if strength == 0:
        return samples.astype(np.float32, copy=True)

    # Bounded FFT blocks keep long uploads within a small cloud server's memory.
    block_size = sample_rate * 2
    context = _choose_window_size(sample_rate)
    processed_channels = [
        np.empty(samples.shape[0], dtype=np.float32) for _ in range(samples.shape[1])
    ]
    for start in range(0, samples.shape[0], block_size):
        end = min(start + block_size, samples.shape[0])
        left = max(0, start - context)
        right = min(samples.shape[0], end + context)
        for channel, output in enumerate(processed_channels):
            cleaned = _spectral_gate(samples[left:right, channel], sample_rate, strength)
            output[start:end] = cleaned[start - left:end - left]
    processed = np.stack(processed_channels, axis=1)

    peak = np.max(np.abs(processed))
    if peak > 1.0:
        processed = processed / peak
    return processed.astype(np.float32)


class StreamingNoiseReducer:
    """Persistent speech denoiser; finish() emits buffered tail samples on stop."""

    def __init__(self, sample_rate: int, hum_filter: str = "off"):
        from .speech import SpeechDenoiser
        self.reducer = SpeechDenoiser(sample_rate, hum_filter=hum_filter)

    def process(self, samples: np.ndarray, strength: float) -> np.ndarray:
        return self.reducer.process(samples, strength)[:, 0]

    def finish(self) -> np.ndarray:
        return self.reducer.finish()[:, 0]


def _validate_audio_size(frames: int, sample_rate: int, channels: int) -> None:
    if not 8000 <= sample_rate <= MAX_SAMPLE_RATE:
        raise AudioError("Supported sample rates range from 8000 to 48000 Hz.")
    if not 1 <= channels <= MAX_CHANNELS:
        raise AudioError("Please upload mono or stereo audio.")
    if frames <= 0:
        raise AudioError("The audio file is empty.")
    if frames > sample_rate * MAX_AUDIO_SECONDS:
        raise AudioError(f"Audio must be no longer than {MAX_AUDIO_SECONDS} seconds.")


def _pcm_to_float32(pcm: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        raw = np.frombuffer(pcm, dtype=np.uint8).astype(np.float32)
        return (raw - 128.0) / 128.0

    if sample_width == 2:
        raw = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        return raw / 32768.0

    if sample_width == 3:
        raw = np.frombuffer(pcm, dtype=np.uint8)
        if raw.size % 3 != 0:
            raise AudioError("The 24-bit WAV data is malformed.")
        triples = raw.reshape(-1, 3).astype(np.int32)
        values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        sign_bit = 1 << 23
        values = (values ^ sign_bit) - sign_bit
        return values.astype(np.float32) / 8388608.0

    if sample_width == 4:
        raw = np.frombuffer(pcm, dtype="<i4").astype(np.float32)
        return raw / 2147483648.0

    raise AudioError("Unsupported WAV bit depth.")


def _float_planar_frame_to_samples(frame: av.AudioFrame) -> np.ndarray:
    array = frame.to_ndarray()
    channels = len(frame.layout.channels) if frame.layout else 1

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] != channels:
        array = array.reshape(channels, -1)

    return np.ascontiguousarray(array.T, dtype=np.float32)


def _extension_from_filename(filename: str | None) -> str:
    if not filename:
        return ""
    return Path(filename).suffix.lower()


def _format_name(filename: str | None, content_type: str | None) -> str:
    extension = _extension_from_filename(filename).lstrip(".")
    if extension:
        return extension
    if content_type:
        return content_type.split("/")[-1].lower()
    return "audio"


def _looks_like_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _spectral_gate(channel: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    if channel.size == 0:
        return channel.astype(np.float32)

    window_size = _choose_window_size(sample_rate)
    hop = window_size // 4
    original_length = channel.size

    pad_mode = "reflect" if channel.size > 1 else "edge"
    padded = np.pad(channel.astype(np.float32), (window_size, window_size), mode=pad_mode)
    frames = _frame_signal(padded, window_size, hop)
    window = np.hanning(window_size).astype(np.float32)
    windowed = frames * window

    spectrum = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(spectrum)
    phase = np.exp(1j * np.angle(spectrum))

    frame_energy = np.sqrt(np.mean(frames * frames, axis=1))
    quiet_frame_count = max(4, int(frame_energy.size * 0.18))
    quiet_indices = np.argsort(frame_energy)[:quiet_frame_count]

    noise_floor = np.percentile(magnitude[quiet_indices], 65, axis=0)
    threshold = noise_floor * (1.2 + strength * 2.6)
    threshold = np.maximum(threshold, 1e-8)

    ratio = magnitude / threshold
    soft_mask = np.clip((ratio - 0.35) / 0.85, 0.0, 1.0)
    min_gain = 1.0 - (strength * 0.92)
    gain = min_gain + (1.0 - min_gain) * (soft_mask**2)

    cleaned_spectrum = magnitude * gain * phase
    cleaned_frames = np.fft.irfft(cleaned_spectrum, n=window_size, axis=1).astype(np.float32)
    cleaned = _overlap_add(cleaned_frames * window, window, hop)

    start = window_size
    end = start + original_length
    return cleaned[start:end].astype(np.float32)


def _choose_window_size(sample_rate: int) -> int:
    target = max(512, int(sample_rate * 0.046))
    power = 1 << (target - 1).bit_length()
    return int(np.clip(power, 512, 4096))


def _frame_signal(signal: np.ndarray, window_size: int, hop: int) -> np.ndarray:
    frame_count = 1 + max(0, int(np.ceil((signal.size - window_size) / hop)))
    needed = (frame_count - 1) * hop + window_size
    if needed > signal.size:
        signal = np.pad(signal, (0, needed - signal.size), mode="constant")

    shape = (frame_count, window_size)
    strides = (signal.strides[0] * hop, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides).copy()


def _overlap_add(frames: np.ndarray, window: np.ndarray, hop: int) -> np.ndarray:
    output_length = (frames.shape[0] - 1) * hop + frames.shape[1]
    output = np.zeros(output_length, dtype=np.float32)
    weights = np.zeros(output_length, dtype=np.float32)
    window_power = window * window

    for index, frame in enumerate(frames):
        start = index * hop
        stop = start + frames.shape[1]
        output[start:stop] += frame
        weights[start:stop] += window_power

    return output / np.maximum(weights, 1e-8)
