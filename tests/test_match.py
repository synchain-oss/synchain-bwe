import numpy as np
import soxr

from conftest import noise_from
from synchain_bwe.dsp import N, RATE, stft
from synchain_bwe.match import match


def frame_db(x, lo, hi):
    S = np.abs(stft(x)) ** 2
    f = np.fft.rfftfreq(N, 1 / RATE)
    return 10 * np.log10(S[:, (f >= lo) & (f < hi)].mean(1) + 1e-30)


def setup(guess_db):
    orig = soxr.resample(noise_from(22050, 3, seed=3), 22050, RATE, quality="VHQ").astype(np.float64)
    guess = noise_from(RATE, 3, seed=4).astype(np.float64)[: len(orig)] * 10 ** (guess_db / 20)
    return orig, guess


def test_edge_level_follows_the_input():
    orig, guess = setup(-15)
    y, st = match(orig, guess, 10000, (8500, 10000))
    d = frame_db(y, 8500, 10000)[20:-20] - frame_db(orig, 8500, 10000)[20:-20]
    assert abs(np.median(d)) < 1.0
    assert st["frames_at_max_up"] == 0


def test_max_up():
    orig, guess = setup(-60)
    y, st = match(orig, guess, 10000, (8500, 10000), max_up_db=24)
    d = frame_db(y, 8500, 10000)[20:-20] - frame_db(orig, 8500, 10000)[20:-20]
    assert st["frames_at_max_up"] > 0.9 * st["frames"]
    assert -40 < np.median(d) < -30


def test_roll_off():
    orig, guess = setup(0)
    y, _ = match(orig, guess, 10000, (8500, 10000))
    assert np.median(frame_db(y, 20500, 24000) - frame_db(y, 12000, 14000)) < -60


def test_ceiling_holds():
    orig, guess = setup(0)
    y, st = match(orig, guess, 10000, (8500, 10000), ceiling=[(11250, -10.0), (22000, -10.0)])
    over = frame_db(y, 12000, 14000)[20:-20] - frame_db(orig, 8500, 10000)[20:-20] + 10
    assert np.mean(over > 3) < 0.02
    assert st["frames_capped"] > 0
