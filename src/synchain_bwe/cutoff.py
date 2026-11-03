"""Finding where a band-limited input stops.

The long-term spectrum is taken in 100 Hz bands (4096-sample Hann frames, half overlapping, silent frames left out). The
cutoff is the steepest fall in it: the 1 kHz below a boundary against the 1 kHz from 200 Hz above it. It counts as a
cutoff when that fall is at least MIN_DROP_DB; a full-band recording falls by a dB or two, a resampled one by far more.

Boundaries inside the file's own band (their upper window ending at least 1 kHz below the file's Nyquist frequency) are
tried first, so that the end of the file's own band, for example the 16-bit noise floor stopping at 22.05 kHz in a
44.1 kHz file, is not taken for the end of the voice. If none of them falls far enough and the file's Nyquist frequency
is under 20 kHz, the file's own band end is the cutoff (a voice made at 22.05 kHz and kept at that rate), and every
boundary is tried. A file whose own band reaches 20 kHz and has no fall inside it has no cutoff.

From the cutoff the crossover start is placed below where the input is still full: the spectrum is followed upward from
1 kHz below the boundary to the first 100 Hz band 6 dB under the level 2 to 1 kHz below the boundary, and the crossover
starts at the last multiple of 500 Hz at least 200 Hz below that band.

Examples, for noise resampled to 48 kHz: from 16 kHz the crossover starts at 7000 Hz, from 22.05 kHz at 10000 Hz, from
24 kHz at 11000 Hz, from 32 kHz at 15000 Hz.
"""
import numpy as np

from .dsp import RATE

MIN_DROP_DB = 20.0
STEP = 500
NYQUIST_MARGIN = 1000.0
FULL_BAND_HZ = 20000.0


def _frames(x, n, h, block=256):
    w = np.hanning(n + 1)[:n]
    starts = range(0, len(x) - n + 1, h)
    for i in range(0, len(starts), block):
        yield np.abs(np.fft.rfft(np.stack([x[k:k + n] for k in starts[i:i + block]]) * w, axis=1)) ** 2


def spectrum_100hz(x):
    """Long-term power spectrum in 100 Hz bands from 0 to 24 kHz, dB. Two passes over the frames, so memory does not
    grow with the length of the file: the first finds the loudest frame, the second averages the frames that are not
    silent (more than 50 dB under the loudest)."""
    n, h = 4096, 2048
    x = np.pad(np.asarray(x, dtype=np.float64), (0, max(0, n - len(x))))
    f = np.fft.rfftfreq(n, 1 / RATE)
    top = max((p.sum(1).max() for p in _frames(x, n, h)), default=0.0)
    acc, count = np.zeros(n // 2 + 1), 0
    for p in _frames(x, n, h):
        keep = p.sum(1) > top * 1e-5 if top > 0 else np.ones(len(p), bool)
        acc += p[keep].sum(0)
        count += int(keep.sum())
    acc /= max(count, 1)
    return np.array([10 * np.log10(acc[(f >= b * 100) & (f < b * 100 + 100)].mean() + 1e-30) for b in range(RATE // 200)])


def _steepest(L, last):
    drops = [(L[b - 10:b].mean() - L[b + 2:b + 12].mean(), b) for b in range(30, last + 1)]
    return max(drops) if drops else None


def detect(x, sr=None):
    """(crossover start in Hz or None, details). x: mono at 48 kHz; sr: the file's own sample rate, if it had another."""
    L = spectrum_100hz(x)
    nyquist = min(sr or RATE, RATE) / 2
    found = _steepest(L, min(len(L) - 12, int((nyquist - NYQUIST_MARGIN) // 100) - 12))
    if not (found and found[0] >= MIN_DROP_DB) and nyquist < FULL_BAND_HZ:
        found = _steepest(L, len(L) - 12)
    if not found:
        return None, {"drop_db": 0.0, "boundary_hz": None}
    drop, b = found
    info = {"drop_db": round(float(drop), 1), "boundary_hz": b * 100}
    if not drop >= MIN_DROP_DB:  # also true for NaN
        return None, info
    ref = L[b - 20:b - 10].mean()
    k = b - 10
    while k < len(L) and L[k] >= ref - 6:
        k += 1
    info["minus_6db_hz"] = k * 100
    start = (k * 100 - 200) // STEP * STEP
    return int(start), info
