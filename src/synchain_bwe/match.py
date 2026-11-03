"""按原声电平 (matched to the input's own level): the model's guess, re-levelled frame by frame.

The model hears only up to 8 kHz and guesses the rest. In the voices this was made for, its guess came out about 15 dB
too quiet, and above 16 kHz it was mostly a steady hiss. So only the shape of the guess is used:

1. Level: in every frame (512 samples, 128 apart) the guess is scaled so that its edge band (the 1.5 kHz just below the
   crossover, which the input really has) matches the input's edge band in that frame. The guess's own edge level is
   averaged over 9 frames first, so that mostly its shape remains. At most +24 dB.
2. Roll-off: from 16 to 20 kHz the result is faded to nothing, which removes the model's hiss up there.
3. Ceiling (when a reference is given): above the crossover, every stretch of the spectrum (averaged over +-8 bins,
   +-750 Hz) is held to the ceiling times the input's edge level in the same frame. The frames are then put back
   together, and because each frame spreads into its neighbours when they are, the result is measured and capped again,
   four times over.

Below the crossover the result is not used: process.py keeps the input there.
"""
import numpy as np

from .dsp import H, N, RATE, istft, smooth_bins, smoothstep, stft, PAD

GUESS_AVG = 9
CAP_SPAN = 8
MAX_UP_DB = 24.0
RECAP = 4
ROLL = (16000.0, 20000.0)


def ceiling_curve(f, ceiling):
    """Linear power ceiling at frequencies f, from [(centre Hz, dB)]."""
    c, d = zip(*ceiling)
    return 10 ** (np.interp(f, c, d) / 10)


def match(orig, guess, xo, edge, ceiling=None, max_up_db=MAX_UP_DB, roll=ROLL, recap=RECAP, count=None):
    """orig, guess: mono at 48 kHz, same length. Returns (matched, stats). count: (start, end) in samples of orig; only
    frames centred there are counted in stats (a chunk's core, not its context). Default: all of orig."""
    op, gp = np.pad(orig, PAD), np.pad(guess, PAD)
    O, G = stft(op), stft(gp)
    f = np.fft.rfftfreq(N, 1 / RATE)
    e = (f >= edge[0]) & (f < edge[1])
    tiny = 1e-14
    po = (np.abs(O[:, e]) ** 2).mean(1)
    pg = np.convolve((np.abs(G[:, e]) ** 2).mean(1), np.ones(GUESS_AVG) / GUESS_AVG, mode="same")
    g = np.sqrt(po / (pg + tiny))
    speech = po > po.max() * 1e-6 if po.max() > 0 else np.zeros(len(po), bool)
    centre = np.arange(len(po)) * H - N // 2 - PAD  # in samples of orig
    lo, hi_ = (0, len(orig)) if count is None else count
    speech &= (centre >= lo) & (centre < hi_)
    up = 10 ** (max_up_db / 20)
    stats = {"frames": int(speech.sum()), "frames_at_max_up": int((g[speech] >= up).sum())}
    g = np.minimum(g, up)
    M = G * g[:, None]
    hi = f >= xo
    if ceiling is not None:
        limit = po[:, None] * ceiling_curve(f[hi], ceiling)[None, :] + tiny
        cap = np.minimum(1.0, np.sqrt(limit / (smooth_bins(np.abs(M[:, hi]) ** 2, CAP_SPAN) + tiny)))
        stats["frames_capped"] = int((cap[speech] < 0.999).any(axis=1).sum())
        M[:, hi] *= cap
    M *= (1 - smoothstep(f, *roll))[None, :]
    y = istft(M, len(op))
    if ceiling is not None:
        for _ in range(recap):
            S = stft(y)
            S[:, hi] *= np.minimum(1.0, np.sqrt(limit / (smooth_bins(np.abs(S[:, hi]) ** 2, CAP_SPAN) + tiny)))
            y = istft(S, len(op))
    return y[PAD:PAD + len(orig)], stats
