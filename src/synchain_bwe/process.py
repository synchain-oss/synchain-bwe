"""One signal through the whole chain.

The input is resampled to 48 kHz (soxr, very high quality). The model guesses the full band; the guess is re-levelled
(match.py) unless the raw guess was asked for; what it adds above the crossover replaces what the input has there:

    added = highpass(guess) - highpass(input)        crossover rising from xo to xo + 500 Hz
    output = input + fade_ends(added)                (or only fade_ends(added), for the band alone)

so below xo the output is the input, sample for sample, and the added band fades in and out over 20 ms at the file's
ends. Long files are done in chunks of about 30 s, each with about 1 s of context on both sides, joined with 250 ms
crossfades. Both the model and the matching only look a fraction of a second around each sample, and every chunk starts
on a multiple of 512 samples from the file's start, so the model's frames (512 apart) and the matching's (128 apart)
fall where they would in one pass over the whole file. Channels are done one by one.
"""
import numpy as np
import soxr

from . import ceiling as ceil
from . import cutoff as cut
from .dsp import RATE, band_db, below_db, fade_ends, highpass
from .match import ROLL, match

XO_WIDTH = ceil.XO_WIDTH
CHUNK_S, CONTEXT_S, JOIN_S = 30.0, 1.0, 0.25
ALIGN = 512  # the model's hop; the matching's (128) divides it
MIN_XO, LOW_XO = 3000, 7000
MIN_SECONDS = 0.1  # shorter than this, the 20 ms fades at the ends cover much of the file
LEAK_LIMIT_DB = -90.0  # the added band more than XO_WIDTH below the crossover, against the input there


class NothingToDo(Exception):
    pass


def resolve_crossover(x48, xo=None, sr=None):
    """(crossover start, how it was found). sr: the input's own sample rate."""
    if xo is not None:
        return int(xo), {"given": True}
    found, info = cut.detect(x48, sr)
    if found is None:
        raise NothingToDo(f"no clear cutoff (steepest fall {info['drop_db']} dB): the input seems to have its high band already")
    return found, info


def check_crossover(xo, roll=ROLL):
    if xo + XO_WIDTH > roll[0]:
        raise NothingToDo(f"the input already reaches {xo / 1000:g} kHz; above {roll[0] / 1000:g} kHz the band is faded out anyway")
    if xo < MIN_XO:
        raise ValueError(f"crossover {xo} Hz is too low; the model needs the input to reach at least {MIN_XO} Hz")


def _context():
    return int(CONTEXT_S * RATE) // ALIGN * ALIGN


def _joins(L):
    """Chunk cores [(start, end)] covering 0..L, inner edges on multiples of ALIGN."""
    c = int(CHUNK_S * RATE)
    if L <= c + 2 * _context():
        return [(0, L)]
    n = int(np.ceil(L / c))
    edges = sorted({0, L, *(round(k * L / n / ALIGN) * ALIGN for k in range(1, n))})
    return list(zip(edges[:-1], edges[1:]))


def _weight(L, a, b, s, e):
    """Over [s, e), the crossfade weight of the chunk whose core is [a, b): 1 in the core, raised-cosine ramps of JOIN_S
    centred on inner boundaries. Neighbouring chunks' weights sum to 1."""
    j = int(JOIN_S * RATE)
    t = np.arange(s, e)

    def r(u):
        return np.where(u <= 0, 0.0, np.where(u >= j, 1.0, 0.5 - 0.5 * np.cos(np.pi * u / j)))

    w = np.ones(e - s)
    if a > 0:
        w *= r(t - (a - j // 2) + 0.5)
    if b < L:
        w *= 1 - r(t - (b - j // 2) + 0.5)
    return w


def added_band(orig, model_in, guesser, xo, ceiling=None, mode="matched", roll=ROLL):
    """What goes on top of orig (mono, 48 kHz), before the end fades, and the matching's frame counts."""
    L = len(orig)
    edge = ceil.edge_band(xo)
    ctx = _context()
    out = np.zeros(L)
    stats = {"frames": 0, "frames_at_max_up": 0}
    joins = _joins(L)
    for a, b in joins:
        s, e = max(0, a - ctx), min(L, b + ctx)
        # The last chunk gets the model input to its own end, which can run a few samples past orig's.
        g = guesser(model_in[s:] if e == L else model_in[s:e], e - s)
        o = orig[s:e]
        if mode == "raw":
            src = g
        else:
            src, st = match(o, g, xo, edge, ceiling, roll=roll, count=(a - s, b - s))
            for k, v in st.items():
                stats[k] = stats.get(k, 0) + v
        piece = highpass(src, xo, xo + XO_WIDTH) - highpass(o, xo, xo + XO_WIDTH)
        if len(joins) == 1:
            out = piece
        else:
            out[s:e] += piece * _weight(L, a, b, s, e)
    return out, stats


def process_channel(x, sr, guesser, xo, ceiling=None, mode="matched", band_only=False, roll=ROLL):
    """x: mono float32 at sr. Returns (48 kHz output, report)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if len(x) < MIN_SECONDS * sr:
        raise NothingToDo(f"shorter than {MIN_SECONDS:g} s")
    if not np.isfinite(x).all():
        raise ValueError("the input has NaN or infinite samples")
    orig = soxr.resample(x, sr, RATE, quality="VHQ").astype(np.float64)
    model_in = guesser.model_input(x, sr)
    layer, stats = added_band(orig, model_in, guesser, xo, ceiling, mode, roll)
    layer = fade_ends(layer)
    out = orig + layer
    edge = ceil.edge_band(xo)
    report = {
        "frames_at_max_up": round(stats["frames_at_max_up"] / max(stats["frames"], 1), 4) if mode != "raw" else None,
        "frames_capped": round(stats["frames_capped"] / max(stats["frames"], 1), 4) if "frames_capped" in stats else None,
        "step_db_before": round(band_db(orig, edge[1] - 500, edge[1]) - band_db(orig, xo + 500, xo + 1000), 1),
        "step_db_after": round(band_db(out, edge[1] - 500, edge[1]) - band_db(out, xo + 500, xo + 1000), 1),
        "added_below_crossover_db": round(below_db(layer, orig, xo), 1),
        "added_500hz_below_crossover_db": round(below_db(layer, orig, xo - XO_WIDTH), 1),
    }
    # Just under the crossover the end fades spread the band a little (about -88 dB in the worst short, noisy file);
    # further down there should be nothing at all.
    if not report["added_500hz_below_crossover_db"] <= LEAK_LIMIT_DB:  # also true for NaN
        raise RuntimeError(f"the added band has content more than {XO_WIDTH:g} Hz below the crossover "
                           f"({report['added_500hz_below_crossover_db']} dB); this is a bug")
    return (layer if band_only else out), report
