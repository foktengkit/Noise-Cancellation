from html.parser import HTMLParser
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.denoise import read_audio_bytes, write_wav_bytes
from scripts.create_sample import CASES, create_samples


ROOT = Path(__file__).resolve().parents[1]


class SampleOptions(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.options = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "option" and "data-url" in attrs:
            self.options.append(attrs)


class SampleTests(unittest.TestCase):
    def test_all_six_public_demos_are_referenced_distinct_and_valid(self):
        options = SampleOptions((ROOT / "frontend/index.html").read_text(encoding="utf-8")).options
        denoising = [option for option in options if option["data-engine"] == "rnnoise"]
        self.assertEqual(len(denoising), 6)
        self.assertEqual({option["data-url"].split("/")[-1] for option in denoising}, {case[0] for case in CASES})
        hashes = set()
        with TestClient(app) as client:
            for option in options:
                with self.subTest(sample=option["value"]):
                    response = client.get(option["data-url"])
                    self.assertEqual(response.status_code, 200)
                    self.assertLess(len(response.content), 1024 * 1024)
                    audio = read_audio_bytes(response.content, option["data-filename"])
                    self.assertEqual(audio.channels, 1)
                    self.assertGreater(len(audio.samples) / audio.sample_rate, 2)
                    self.assertLessEqual(len(audio.samples) / audio.sample_rate, 20)
                    self.assertTrue(np.all(np.isfinite(audio.samples)))
                    self.assertGreater(float(np.max(np.abs(audio.samples))), .01)
                    self.assertLess(float(np.max(np.abs(audio.samples))), .999)
                    hashes.add(hashlib.sha256(response.content).hexdigest())
        self.assertEqual(len(hashes), 7)

    def test_generation_is_repeatable_and_uses_only_the_explicit_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "synthetic.wav"
            t = np.arange(16000) / 16000
            clean.write_bytes(write_wav_bytes(.15 * np.sin(2 * np.pi * 300 * t), 16000))
            first = create_samples(clean, root / "first")
            second = create_samples(clean, root / "second")
            self.assertEqual(len(first), 6)
            self.assertEqual([path.read_bytes() for path in first], [path.read_bytes() for path in second])
            self.assertEqual(len({path.read_bytes() for path in first}), 6)
            control = read_audio_bytes((root / "first/sample-clean.wav").read_bytes(), "clean.wav")
            np.testing.assert_array_equal(control.samples[:16000], 0)
            np.testing.assert_array_equal(control.samples[-16000:], 0)

    def test_silent_or_wrong_format_references_do_not_publish_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / "synthetic.wav"
            for rate, signal in ((16000, np.zeros(16000)), (8000, np.ones(8000) * .1)):
                clean.write_bytes(write_wav_bytes(signal, rate))
                with self.assertRaises(ValueError):
                    create_samples(clean, root / "public")
                self.assertFalse((root / "public").exists())


if __name__ == "__main__":
    unittest.main()
