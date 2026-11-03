import numpy as np
import pytest
import soxr

from conftest import noise_from, speechlike
from synchain_bwe import ceiling


def test_bands_for_10k():
    assert ceiling.bands_above(10000) == [(10500, 12000), (12000, 14000), (14000, 16000), (16000, 18000), (18000, 20000), (20000, 24000)]
    assert ceiling.edge_band(10000) == (8500, 10000)


def test_bands_for_7k_and_11k():
    assert ceiling.bands_above(7000)[0] == (7500, 10000)
    assert ceiling.bands_above(11000)[:2] == [(11500, 14000), (14000, 16000)]


def test_white_reference_is_flat():
    c = ceiling.from_reference(noise_from(48000, 8).astype(np.float64), 10000)
    for centre, db in c["ceiling"]:
        assert 0 <= db < 4, (centre, db)  # equal power per hertz: the 90th percentile of the ratio sits a little above 0 dB


def test_tilted_reference_falls():
    c = ceiling.from_reference(speechlike(48000, 10).astype(np.float64), 10000)
    dbs = [d for _, d in c["ceiling"]]
    assert all(a > b for a, b in zip(dbs, dbs[1:]))


def test_band_limited_reference_is_refused():
    x = soxr.resample(noise_from(22050, 6), 22050, 48000).astype(np.float64)
    with pytest.raises(ValueError, match="stops at"):
        ceiling.from_reference(x, 10000)


def test_save_load(tmp_path):
    c = ceiling.from_reference(noise_from(48000, 6).astype(np.float64), 10000)
    ceiling.save(c, tmp_path / "c.json")
    back = ceiling.load(tmp_path / "c.json")
    assert back["crossover_hz"] == 10000 and back["ceiling"] == [tuple(x) for x in c["ceiling"]]
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(ValueError):
        ceiling.load(tmp_path / "bad.json")


def test_band_limited_44k_reference_is_refused():
    y = soxr.resample(noise_from(32000, 8), 32000, 44100, quality="VHQ")
    y = np.round(y * 32767) / 32767
    x = soxr.resample(y, 44100, 48000, quality="VHQ").astype(np.float64)
    with pytest.raises(ValueError, match="stops at"):
        ceiling.from_reference(x, 10000, 44100)


def test_low_rate_and_nan_references_are_refused():
    x = noise_from(48000, 8).astype(np.float64)
    with pytest.raises(ValueError, match="44.1 kHz"):
        ceiling.from_reference(x, 10000, 32000)
    x[5] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        ceiling.from_reference(x, 10000)


def test_gently_dulled_reference_is_refused():
    x = noise_from(48000, 8).astype(np.float64)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / 48000)
    t = np.clip((f - 10500) / 6000, 0, 1)
    X *= 10 ** (-60 * t * t * (3 - 2 * t) / 20)  # down 60 dB over 6 kHz: never 20 dB in a step, so no cutoff is found
    with pytest.raises(ValueError, match="almost nothing"):
        ceiling.from_reference(np.fft.irfft(X, len(x)), 10000)


@pytest.mark.parametrize("body", [
    '{"synchain_bwe_ceiling": 1, "crossover_hz": 10000, "ceiling": [[11250, NaN]]}',
    '{"synchain_bwe_ceiling": 1, "crossover_hz": 10000, "ceiling": [[13000, -3], [11250, -2]]}',
    '{"synchain_bwe_ceiling": 1, "crossover_hz": 10000, "ceiling": []}',
    '{"synchain_bwe_ceiling": 1, "ceiling": [[11250, -2]]}',
    '{"synchain_bwe_ceiling": 1, "crossover_hz": 10000, "ceiling": [["x", -2]]}',
])
def test_broken_ceiling_files_are_refused(tmp_path, body):
    (tmp_path / "c.json").write_text(body)
    with pytest.raises(ValueError):
        ceiling.load(tmp_path / "c.json")


def test_short_reference_is_refused():
    with pytest.raises(ValueError, match="too little speech"):
        ceiling.from_reference(noise_from(48000, 3).astype(np.float64), 10000)


@pytest.mark.parametrize("xo", [3000, 7000, 11000, 15000])
def test_full_band_voice_is_a_reference_at_any_crossover(xo):
    c = ceiling.from_reference(speechlike(48000, 12).astype(np.float64), xo)
    assert np.isfinite([d for _, d in c["ceiling"]]).all()


@pytest.mark.parametrize("body", ["[]", '{"synchain_bwe_ceiling": 1, "crossover_hz": Infinity, "ceiling": [[11250, -2]]}'])
def test_odd_ceiling_files_are_refused(tmp_path, body):
    (tmp_path / "c.json").write_text(body)
    with pytest.raises(ValueError):
        ceiling.load(tmp_path / "c.json")
