import numpy as np

from synchain_bwe.dsp import RATE, below_db, fade_ends, highpass, istft, stft


def sine(f, seconds=1.0, a=0.5):
    return a * np.sin(2 * np.pi * f * np.arange(int(RATE * seconds)) / RATE)


def test_highpass_keeps_above_and_removes_below():
    lo = highpass(sine(9000), 10000, 10500)
    hi = highpass(sine(11000), 10000, 10500)
    assert 20 * np.log10(np.abs(lo[5000:-5000]).max() / 0.5) < -100  # the abrupt ends of the test tone do pass
    assert np.allclose(hi[2000:-2000], sine(11000)[2000:-2000], atol=1e-6)


def test_highpass_does_not_wrap():
    x = np.zeros(RATE)
    x[-1] = 1.0  # a click at the very end must not show up at the start
    assert np.abs(highpass(x, 10000, 10500)[:2000]).max() < 1e-9


def test_stft_round_trip():
    x = np.random.default_rng(1).standard_normal(10000)
    assert np.abs(istft(stft(x), len(x)) - x).max() < 1e-10


def test_fade_ends():
    y = fade_ends(np.ones(5000))
    assert y[0] < 1e-5 and y[-1] < 1e-5 and np.allclose(y[960:-960], 1) and np.allclose(y, y[::-1])
    assert len(fade_ends(np.ones(10))) == 10


def test_below_db_measures_only_below():
    layer = sine(10500, 3.0) * np.hanning(3 * RATE)  # smooth all through, so it has nothing below 10 kHz itself
    ref = np.random.default_rng(2).standard_normal(len(layer)) * 0.1
    assert below_db(layer, ref, 10000) < -100
    assert abs(below_db(ref, ref, 10000)) < 1e-9
