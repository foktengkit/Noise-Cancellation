"""Disk-backed, private live recordings with bounded PCM conversion buffers."""

import tempfile
import wave

import numpy as np

from .denoise import AudioError, _validate_audio_size, write_wav_frames


class WavSpool:
    def __init__(self, sample_rate, channels=1):
        _validate_audio_size(1, sample_rate, channels)
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames = 0
        self.file = tempfile.TemporaryFile(mode="w+b")
        self.writer = None
        try:
            self.writer = wave.open(self.file, "wb")
            self.writer.setnchannels(channels)
            self.writer.setsampwidth(2)
            self.writer.setframerate(sample_rate)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def write(self, samples):
        if self.writer is None:
            raise AudioError("The recording is already finished.")
        samples = np.asarray(samples)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or samples.shape[1] != self.channels or not np.isfinite(samples).all():
            raise AudioError("Invalid recording samples.")
        if not len(samples):
            return
        _validate_audio_size(self.frames + len(samples), self.sample_rate, self.channels)
        write_wav_frames(self.writer, samples)
        self.frames += len(samples)

    def read_wav(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.file.seek(0)
        return self.file.read()

    def close(self):
        try:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
        finally:
            self.file.close()
