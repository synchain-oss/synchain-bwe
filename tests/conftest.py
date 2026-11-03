import numpy as np
import pytest
import soxr

RATE = 48000


class FakeGuesser:
    """Stands in for the model, so the tests need no torch. Like the model it hears the input through 16 kHz, and it
    makes a deterministic, sample-local full-band guess (the 16 kHz copy plus its square, which reaches up to 16 kHz, plus
    a cube, up to 24 kHz), so chunked and whole-file runs can be compared."""

    def model_input(self, x, sr):
        y = soxr.resample(np.asarray(x, np.float32), sr, 16000, quality="VHQ")
        return soxr.resample(y, 16000, RATE, quality="VHQ").astype(np.float32)

    def __call__(self, segment, length=None):
        s = np.asarray(segment, np.float64)
        g = s + 0.5 * s * np.abs(s) + 2.0 * s ** 3
        length = len(s) if length is None else length
        return np.pad(g, (0, max(0, length - len(g))))[:length]


@pytest.fixture
def fake():
    return FakeGuesser()


def noise_from(sr, seconds, seed=0, tilt=0.0):
    """White noise made at sr, or tilted: power falling `tilt` dB per octave above 1 kHz."""
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    x = rng.standard_normal(n)
    if tilt:
        X = np.fft.rfft(x)
        f = np.maximum(np.fft.rfftfreq(n, 1 / sr), 50)
        X *= (f / 1000) ** (-tilt / 20 / np.log10(2))
        x = np.fft.irfft(X, n)
    return (0.1 * x / np.std(x)).astype(np.float32)


def speechlike(sr, seconds, seed=0):
    """Tilted noise in bursts with gaps, made at sr."""
    x = noise_from(sr, seconds, seed, tilt=6.0).astype(np.float64)
    t = np.arange(len(x)) / sr
    env = (np.sin(2 * np.pi * 1.7 * t) > -0.2).astype(float)
    k = int(0.01 * sr)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    return (x * env).astype(np.float32)
