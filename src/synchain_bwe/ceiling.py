"""The ceiling: how bright a real voice gets above the cutoff, measured on a full-band reference recording.

For each band above the crossover, per hertz, against the same frame's edge band (the 1.5 kHz just below the crossover):
the level the reference exceeds in only 10% of its speech frames. 512-sample Hann frames, 128 apart; speech frames are
those whose 0.1 to 8 kHz power is within 40 dB of the recording's loudest frame. The bands run from the crossover's top
to the first even kHz at least 1 kHz above it, then 2 kHz wide up to 20 kHz, then 20 to 24 kHz; with the crossover at
10 kHz that is 10.5-12, 12-14, 14-16, 16-18, 18-20 and 20-24 kHz. Between band centres the ceiling is interpolated;
below the first and above the last it is held.

A reference is refused if it is under 44.1 kHz, stops early (a cutoff is found below 19.5 kHz), has less than
MIN_SPEECH_S of speech, or is dull: in its speech frames, 16 to 18 kHz is more than 45 dB under 8.5 to 10 kHz even in
the brightest tenth.

The ceiling is a few numbers per cutoff. It describes the reference voice, so treat a saved ceiling file as you would
treat a description of that voice.
"""
import json

import numpy as np

from . import cutoff as cut
from .dsp import RATE, N, H

PERCENTILE = 90
MIN_RATE = 44100
MIN_SPEECH_S = 4.0
DULL_BANDS = ((8500.0, 10000.0), (16000.0, 18000.0))
MIN_BRIGHT_DB = -45.0  # 16-18 kHz against 8.5-10 kHz, 90th percentile; a real voice is nowhere near this
EDGE_WIDTH = 1500.0
XO_WIDTH = 500.0


def edge_band(xo):
    return (xo - EDGE_WIDTH, float(xo))


def bands_above(xo):
    lo = xo + XO_WIDTH
    edges = [lo] + [f for f in range(2000, 20001, 2000) if f >= lo + 1000] + [24000]
    return [(float(a), float(b)) for a, b in zip(edges[:-1], edges[1:])]


def from_reference(ref, xo, sr=None):
    """ref: mono at 48 kHz; sr: the recording's own sample rate, if it had another. Returns a dict: crossover, edge,
    bands, ceiling [(centre Hz, dB)], speech frames."""
    if sr is not None and sr < MIN_RATE:
        raise ValueError(f"the reference recording is {sr} Hz; it needs to be 44.1 kHz or more, with its full band")
    if not np.isfinite(ref).all():
        raise ValueError("the reference recording has NaN or infinite samples")
    found, info = cut.detect(ref, sr)
    if found is not None and found < 19500:
        raise ValueError(f"the reference recording itself stops at about {info['minus_6db_hz'] / 1000:g} kHz; "
                         "use a full-band recording (44.1 or 48 kHz, nothing cut below 20 kHz)")
    w = np.hanning(N + 1)[:N]
    x = np.pad(np.asarray(ref, dtype=np.float64), (0, max(0, N - len(ref))))
    f = np.fft.rfftfreq(N, 1 / RATE)
    edge = edge_band(xo)
    bands = bands_above(xo)
    tot, edge_p, band_p = [], [], {b: [] for b in bands}
    dull_p = {b: [] for b in DULL_BANDS}
    starts = range(0, len(x) - N + 1, H)
    for i in range(0, len(starts), 4096):
        fr = np.stack([x[k:k + N] for k in starts[i:i + 4096]]) * w
        P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        tot.append(P[:, (f >= 100) & (f < 8000)].sum(1))
        edge_p.append(P[:, (f >= edge[0]) & (f < edge[1])].mean(1))
        for b in bands:
            band_p[b].append(P[:, (f >= b[0]) & (f < b[1])].mean(1))
        for b in DULL_BANDS:
            dull_p[b].append(P[:, (f >= b[0]) & (f < b[1])].mean(1))
    tot, edge_p = np.concatenate(tot), np.concatenate(edge_p)
    speech = tot > tot.max() * 1e-4
    if speech.sum() * H / RATE < MIN_SPEECH_S:
        raise ValueError(f"the reference has too little speech ({speech.sum() * H / RATE:.1f} s); it needs at least "
                         f"{MIN_SPEECH_S:g} s, and 10 s or more is best")
    mid, high = (np.concatenate(dull_p[b]) for b in DULL_BANDS)
    bright = float(np.percentile(10 * np.log10((high[speech] + 1e-30) / (mid[speech] + 1e-30)), PERCENTILE))
    if not bright >= MIN_BRIGHT_DB:
        raise ValueError(f"the reference recording has almost nothing at 16 to 18 kHz ({bright:.1f} dB against 8.5 to "
                         "10 kHz); use a full-band recording")
    ceiling = []
    for b in bands:
        rel = 10 * np.log10((np.concatenate(band_p[b]) + 1e-30) / (edge_p + 1e-30))
        ceiling.append(((b[0] + b[1]) / 2, round(float(np.percentile(rel[speech], PERCENTILE)), 1)))
    if not np.isfinite([d for _, d in ceiling]).all():
        raise ValueError(f"no ceiling can be measured for a crossover at {xo} Hz")
    return {"crossover_hz": int(xo), "edge_hz": list(edge), "bands_hz": [list(b) for b in bands],
            "ceiling": [list(c) for c in ceiling], "percentile": PERCENTILE, "speech_frames": int(speech.sum())}


def save(c, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"synchain_bwe_ceiling": 1, **c}, fh, indent=1)


def load(path):
    with open(path, encoding="utf-8") as fh:
        c = json.load(fh)
    if not isinstance(c, dict) or c.get("synchain_bwe_ceiling") != 1:
        raise ValueError(f"{path} is not a ceiling file written by synchain-bwe")
    try:
        c["ceiling"] = [(float(a), float(b)) for a, b in c["ceiling"]]
        c["crossover_hz"] = int(c["crossover_hz"])
    except (KeyError, TypeError, ValueError, OverflowError):
        raise ValueError(f"{path} is not a valid ceiling file") from None
    centres = [a for a, _ in c["ceiling"]]
    if not c["ceiling"] or not np.isfinite(np.array(c["ceiling"])).all() or any(a >= b for a, b in zip(centres, centres[1:])):
        raise ValueError(f"{path} is not a valid ceiling file (needs finite values at increasing frequencies)")
    return c
