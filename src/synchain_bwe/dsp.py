"""Signal helpers. Everything here works on mono float64 arrays at 48 kHz."""
import numpy as np

RATE = 48000
PAD = 4800  # zeros on each side before a whole-signal FFT filter, so nothing wraps from one end to the other
FADE = 960  # 20 ms: the added band fades in and out over this at the ends of a file
N, H = 512, 128  # STFT frames for the level matching: 10.7 ms, 2.7 ms apart


def smoothstep(f, lo, hi):
    t = np.clip((np.asarray(f, dtype=np.float64) - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


def highpass(x, lo, hi):
    """x above a crossover that rises from 0 at lo Hz to 1 at hi Hz (smoothstep), zero phase. The signal is padded with
    zeros first, so the circular FFT does not wrap one end into the other."""
    y = np.pad(np.asarray(x, dtype=np.float64), PAD)
    m = smoothstep(np.fft.rfftfreq(len(y), 1 / RATE), lo, hi)
    return np.fft.irfft(np.fft.rfft(y) * m, len(y))[PAD:PAD + len(x)]


def fade_ends(x, n=FADE):
    """Raised-cosine fade in over the first n samples and out over the last n."""
    n = min(n, len(x) // 2)
    w = np.ones(len(x))
    if n:
        r = 0.5 - 0.5 * np.cos(np.pi * (np.arange(n) + 0.5) / n)
        w[:n] = r
        w[len(x) - n:] = r[::-1]
    return x * w


def _window():
    return np.hanning(N + 1)[:N]  # periodic Hann


def stft(x):
    xp = np.pad(x, (N, N + H))
    return np.fft.rfft(np.lib.stride_tricks.sliding_window_view(xp, N)[::H] * _window(), axis=1)


def istft(S, length):
    w = _window()
    frames = np.fft.irfft(S, N, axis=1) * w
    out = np.zeros(N + H * (len(frames) - 1))
    for k, fr in enumerate(frames):
        out[k * H:k * H + N] += fr
    return out[N:N + length] / 1.5  # a periodic Hann squared, a quarter of its length apart, sums to 1.5


def smooth_bins(a, k):
    """Moving average over 2k+1 bins along the last axis, edges held."""
    c = np.cumsum(np.pad(a, ((0, 0), (k + 1, k)), mode="edge"), axis=1)
    return (c[:, 2 * k + 1:] - c[:, :-2 * k - 1]) / (2 * k + 1)


def below_db(layer, ref, top, n=8192, block=128):
    """Power of layer below top Hz against the power of ref below top Hz, in dB, over the whole signal: Hann frames of n
    samples, half overlapping (they sum to a constant), with n zeros added at each end so the ends count fully. A frame
    edge leaks a component 500 Hz away by less than -105 dB. Frames are taken a block at a time to bound memory."""
    w = np.hanning(n + 1)[:n]
    m = np.fft.rfftfreq(n, 1 / RATE) < top
    a = b = 0.0
    lp, rp = np.pad(layer, n), np.pad(ref, n)
    starts = range(0, len(lp) - n + 1, n // 2)
    for i in range(0, len(starts), block):
        s = starts[i:i + block]
        fa = np.stack([lp[k:k + n] for k in s]) * w
        fb = np.stack([rp[k:k + n] for k in s]) * w
        a += (np.abs(np.fft.rfft(fa, axis=1)[:, m]) ** 2).sum()
        b += (np.abs(np.fft.rfft(fb, axis=1)[:, m]) ** 2).sum()
    return float(10 * np.log10(a / (b + 1e-30) + 1e-30))


def band_db(x, lo, hi, nfft=2048, block=512):
    """Mean power per hertz in a band over the whole signal, dB. Only differences between bands mean anything. Frames
    are taken a block at a time to bound memory."""
    hop = nfft // 2
    w = np.hanning(nfft)
    f = np.fft.rfftfreq(nfft, 1 / RATE)
    m = (f >= lo) & (f < hi)
    x = np.pad(x, (0, max(0, nfft + 1 - len(x))))
    starts = range(0, len(x) - nfft + 1, hop)
    acc = 0.0
    for i in range(0, len(starts), block):
        fr = np.stack([x[k:k + nfft] for k in starts[i:i + block]]) * w
        acc += (np.abs(np.fft.rfft(fr, axis=1))[:, m] ** 2).sum()
    return float(10 * np.log10(acc / (len(starts) * m.sum()) + 1e-20))
