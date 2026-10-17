import unittest

import numpy as np

from scripts.evaluate_quality_matrix import NOISE_TYPES, build_noise, make_mixture, measure, power, prepare_reference


class QualityMetricsTests(unittest.TestCase):
    def setUp(self):
        self.rate = 16000
        t = np.arange(self.rate) / self.rate
        source = 0.1 * np.sin(2 * np.pi * 300 * t)
        self.reference, self.active, self.quiet = prepare_reference(source, self.rate)

    def test_reference_rejects_silence_and_nonfinite_audio(self):
        for samples in (np.zeros(100), np.array([np.nan]), np.empty(0)):
            with self.assertRaises(ValueError):
                prepare_reference(samples, self.rate)

    def test_activity_and_quiet_masks_do_not_depend_on_the_output(self):
        self.assertGreater(np.count_nonzero(self.active), 0)
        self.assertFalse(np.any(self.active & self.quiet))
        self.assertEqual(power(self.reference[self.quiet]), 0)

    def test_noise_is_seeded_finite_and_nonempty_for_every_case(self):
        for kind in NOISE_TYPES:
            noise = build_noise(kind, self.reference, self.rate, 420)
            self.assertEqual(noise.shape, self.reference.shape)
            self.assertTrue(np.all(np.isfinite(noise)))
            self.assertGreater(power(noise[self.active]), 0)
            np.testing.assert_array_equal(noise, build_noise(kind, self.reference, self.rate, 420))

    def test_mixture_preserves_requested_snr_without_clipping(self):
        noise = build_noise("white_fan", self.reference, self.rate, 420)
        for snr in (-5, 5, 15):
            clean, noisy, scale = make_mixture(self.reference, noise, self.active, snr)
            actual = 10 * np.log10(power(clean[self.active]) / power((noisy - clean)[self.active]))
            self.assertAlmostEqual(actual, snr, places=5)
            self.assertLessEqual(np.max(np.abs(noisy)), 0.950001)
            np.testing.assert_allclose(clean, self.reference * scale, atol=1e-7)

    def test_muting_is_not_reported_as_successful_speech_enhancement(self):
        metrics = measure(self.reference, np.zeros_like(self.reference), self.active, self.quiet)
        self.assertIsNone(metrics["siSdrDb"])
        self.assertIsNone(metrics["activeSiSdrDb"])
        self.assertTrue(metrics["silentOutput"])
        self.assertLess(metrics["activeRmsGainDb"], -60)

    def test_uniform_gain_is_distinguished_from_speech_distortion(self):
        metrics = measure(self.reference, self.reference * 0.5, self.active, self.quiet)
        self.assertAlmostEqual(metrics["activeRmsGainDb"], -6.021, places=3)
        self.assertGreater(metrics["siSdrDb"], 80)
        self.assertFalse(metrics["silentOutput"])

    def test_invalid_outputs_are_rejected(self):
        for estimate in (self.reference[:-1], np.full_like(self.reference, np.nan),
                         np.full_like(self.reference, 1.1)):
            with self.assertRaises(ValueError):
                measure(self.reference, estimate, self.active, self.quiet)


if __name__ == "__main__":
    unittest.main()
