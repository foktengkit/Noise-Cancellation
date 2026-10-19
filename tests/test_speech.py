import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend.denoise import AudioError, MAX_AUDIO_SECONDS, StreamingNoiseReducer, reduce_noise
from backend.speech import MODEL_PATH, MODEL_SHA256, STRENGTH_EXPONENT, SpeechDenoiser, verify_model


class SpeechTests(unittest.TestCase):
    def test_bundled_model_integrity(self):
        self.assertEqual(hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(), MODEL_SHA256)

    def test_tampered_or_missing_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.rnnn"
            for present in (False, True):
                if present:
                    path.write_bytes(b"not a neural model")
                verify_model.cache_clear()
                with patch("backend.speech.MODEL_PATH", path), self.assertRaises(AudioError):
                    SpeechDenoiser(16000)
        verify_model.cache_clear()

    def test_noise_suppression_is_stronger_at_full_strength(self):
        noise = np.random.default_rng(24).normal(0, .05, (32000, 1)).astype(np.float32)
        soft = reduce_noise(noise, 16000, .5)
        strong = reduce_noise(noise, 16000, 1)
        # Exclude model startup; this is a noise-only test, not a speech-quality claim.
        original_power = np.mean(noise[8000:] ** 2)
        self.assertLess(np.mean(strong[8000:] ** 2), original_power / 100)
        self.assertLess(np.mean(strong ** 2), np.mean(soft ** 2))

    def test_silence_has_no_nan_or_generated_sound(self):
        result = reduce_noise(np.zeros((16017, 2), np.float32), 16000)
        self.assertEqual(result.shape, (16017, 2))
        self.assertLess(np.max(np.abs(result)), 1e-6)

    def test_short_and_partial_frames_preserve_length_at_all_rates(self):
        for rate in (8000, 11025, 16000, 22050, 44100, 48000):
            for channels in (1, 2):
                for length in (1, 127, rate // 4 + 17, rate * 2 + 113):
                    with self.subTest(rate=rate, channels=channels, length=length):
                        samples = np.random.default_rng(length).normal(0, .03, (length, channels)).astype(np.float32)
                        cleaned = reduce_noise(samples, rate)
                        self.assertEqual(cleaned.shape, samples.shape)
                        self.assertTrue(np.all(np.isfinite(cleaned)))
                        self.assertLessEqual(np.max(np.abs(cleaned)), 1)

    def test_zero_strength_stream_preserves_last_sample_exactly(self):
        for rate in (8000, 16000, 44100, 48000):
            samples = np.random.default_rng(rate).normal(0, .04, rate + 127).astype(np.float32)
            samples[-1] = .7
            reducer = StreamingNoiseReducer(rate)
            output = [reducer.process(samples[i:i + 127], 0) for i in range(0, len(samples), 127)]
            output.append(reducer.finish())
            np.testing.assert_array_equal(np.concatenate(output), samples)
            self.assertEqual(reducer.finish().size, 0)
            with self.assertRaises(AudioError):
                reducer.process(samples[:10], 0)

    def test_stream_matches_file_without_chunk_seams(self):
        for rate in (16000, 44100, 48000):
            samples = np.random.default_rng(16).normal(0, .04, rate * 3 + 127).astype(np.float32)
            file_output = reduce_noise(samples, rate, 1)[:, 0]
            reducer = StreamingNoiseReducer(rate)
            outputs = [reducer.process(samples[i:i + rate // 4], 1)
                       for i in range(0, len(samples), rate // 4)]
            outputs.append(reducer.finish())
            np.testing.assert_allclose(np.concatenate(outputs), file_output, atol=1e-6)

    def test_neural_state_is_not_shared_between_users(self):
        a = np.random.default_rng(20).normal(0, .03, 8000).astype(np.float32)
        b = np.random.default_rng(21).normal(0, .2, 8000).astype(np.float32)
        first, second = StreamingNoiseReducer(16000), StreamingNoiseReducer(16000)
        first_output = first.process(a, 1)
        second.process(b, 1)
        first_tail = first.finish()
        second.finish()
        np.testing.assert_array_equal(np.r_[first_output, first_tail], reduce_noise(a, 16000, 1)[:, 0])

    def test_strength_changes_apply_to_corresponding_input_not_delayed_reply(self):
        x = np.random.default_rng(7).normal(0, .04, 8000).astype(np.float32)
        reducer = StreamingNoiseReducer(16000)
        chunks = [reducer.process(x[:4000], 0), reducer.process(x[4000:], 1), reducer.finish()]
        output = np.concatenate(chunks)
        expected_wet = reduce_noise(x, 16000, 1)[:, 0]
        np.testing.assert_array_equal(output[:4000], x[:4000])
        np.testing.assert_allclose(output[4000:], expected_wet[4000:], atol=1e-6)

    def test_intermediate_strength_preserves_aligned_dry_detail(self):
        source = np.random.default_rng(42).normal(0, .04, (16017, 2)).astype(np.float32)
        wet = reduce_noise(source, 16000, 1)
        previous = 0.
        for strength in (.25, .5, .85):
            with self.subTest(strength=strength):
                amount = strength ** STRENGTH_EXPONENT
                self.assertLess(previous, amount)
                self.assertLess(amount, strength)
                expected = np.clip(source*(1-amount) + wet*amount, -1, 1)
                np.testing.assert_allclose(reduce_noise(source, 16000, strength), expected, atol=1e-6)
                previous = amount

    def test_fractional_strength_stream_matches_file_for_irregular_chunks(self):
        source = np.random.default_rng(43).normal(0, .03, 32017).astype(np.float32)
        for size in (127, 4000):
            with self.subTest(size=size):
                reducer = StreamingNoiseReducer(16000)
                output = [reducer.process(source[start:start+size], .85)
                          for start in range(0, len(source), size)]
                output.append(reducer.finish())
                np.testing.assert_allclose(np.concatenate(output),
                                           reduce_noise(source, 16000, .85)[:, 0], atol=1e-6)

    def test_fractional_strength_changes_follow_input_alignment(self):
        source = np.random.default_rng(44).normal(0, .04, 16017).astype(np.float32)
        reducer = StreamingNoiseReducer(16000)
        output = np.concatenate([reducer.process(source[:7001], .25),
                                 reducer.process(source[7001:], .85), reducer.finish()])
        wet = reduce_noise(source, 16000, 1)[:, 0]
        for start, end, strength in ((0, 7001, .25), (7001, len(source), .85)):
            amount = strength ** STRENGTH_EXPONENT
            expected = source[start:end]*(1-amount) + wet[start:end]*amount
            np.testing.assert_allclose(output[start:end], expected, atol=1e-6)

    def test_invalid_samples_strength_and_duration_are_bounded(self):
        for strength in (float("nan"), float("inf"), -1, 2):
            with self.assertRaises(AudioError):
                reduce_noise(np.ones((10, 1)), 16000, strength)
        with self.assertRaises(AudioError):
            reduce_noise(np.array([[float("nan")]]), 16000)
        with self.assertRaises(AudioError):
            SpeechDenoiser(16000).process(np.zeros((32001, 1)), 1)
        reducer = SpeechDenoiser(16000)
        reducer.received = MAX_AUDIO_SECONDS * 16000
        with self.assertRaises(AudioError):
            reducer.process(np.zeros((1, 1)), 1)


if __name__ == "__main__":
    unittest.main()
