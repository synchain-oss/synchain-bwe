import numpy as np
import pytest
import soxr

from conftest import noise_from, speechlike
from synchain_bwe import process
from synchain_bwe.dsp import RATE, band_db, below_db


def run(fake, x, sr, **kw):
    return process.process_channel(x, sr, fake, 10000, **kw)


def test_below_crossover_is_the_input(fake):
    x = speechlike(22050, 4)
    out, rep = run(fake, x, 22050)
    orig = soxr.resample(x, 22050, RATE, quality="VHQ").astype(np.float64)
    assert len(out) == len(orig)
    assert below_db(out - orig, orig, 10000) < -100
    assert rep["step_db_before"] > 10
    assert band_db(out, 12000, 14000) > band_db(orig, 12000, 14000) + 20


def test_band_only_is_the_difference(fake):
    x = speechlike(22050, 3)
    out, _ = run(fake, x, 22050)
    band, _ = run(fake, x, 22050, band_only=True)
    orig = soxr.resample(x, 22050, RATE, quality="VHQ").astype(np.float64)
    assert np.abs(out - orig - band).max() < 1e-12


def test_ends_fade(fake):
    band, _ = run(fake, speechlike(22050, 3), 22050, band_only=True)
    peak = np.abs(band).max()
    assert abs(band[0]) < 1e-4 * peak and abs(band[-1]) < 1e-4 * peak


def test_chunks_join_cleanly(fake, monkeypatch):
    x = speechlike(22050, 9, seed=5)
    whole, _ = run(fake, x, 22050, band_only=True)
    monkeypatch.setattr(process, "CHUNK_S", 2.0)
    monkeypatch.setattr(process, "CONTEXT_S", 0.5)
    assert len(process._joins(len(whole))) > 3
    chunked, _ = run(fake, x, 22050, band_only=True)
    err = 10 * np.log10(((chunked - whole) ** 2).sum() / (whole ** 2).sum())
    assert err < -120, err


def test_join_weights_sum_to_one(monkeypatch):
    monkeypatch.setattr(process, "CHUNK_S", 1.0)
    monkeypatch.setattr(process, "CONTEXT_S", 0.5)
    L = 5 * RATE + 123
    joins = process._joins(L)
    assert len(joins) > 1
    ctx = process._context()
    full = sum(np.pad(process._weight(L, a, b, max(0, a - ctx), min(L, b + ctx)), (max(0, a - ctx), L - min(L, b + ctx))) for a, b in joins)
    assert np.allclose(full, 1.0)
    assert all(a % process.ALIGN == 0 for a, _ in joins)


def test_raw_mode(fake):
    _, rep = run(fake, speechlike(22050, 3), 22050, mode="raw")
    assert rep["frames_at_max_up"] is None


def test_full_band_input_is_skipped():
    with pytest.raises(process.NothingToDo):
        process.resolve_crossover(noise_from(48000, 3).astype(np.float64))
    with pytest.raises(process.NothingToDo):
        process.check_crossover(15600)
    with pytest.raises(ValueError):
        process.check_crossover(2500)


def test_16k_input(fake):
    x = speechlike(16000, 3)
    out, _ = process.process_channel(x, 16000, fake, 7000)
    orig = soxr.resample(x, 16000, RATE, quality="VHQ").astype(np.float64)
    assert below_db(out - orig, orig, 7000) < -100


def test_nan_and_tiny_inputs(fake):
    x = speechlike(22050, 1)
    x[100] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        run(fake, x, 22050)
    with pytest.raises(process.NothingToDo):
        run(fake, speechlike(22050, 1)[:1000], 22050)


def test_band_db_does_not_depend_on_the_block():
    x = noise_from(48000, 3, tilt=6.0).astype(np.float64)
    assert abs(band_db(x, 12000, 14000, block=7) - band_db(x, 12000, 14000)) < 1e-9


@pytest.mark.parametrize("sr,xo", [(22050, 10000), (24000, 11000)])
def test_short_noisy_inputs_are_not_called_a_bug(fake, sr, xo):
    for seconds in (0.11, 0.2, 0.3, 0.5):
        for seed in range(6):
            x = noise_from(sr, seconds, seed=seed)
            _, rep = process.process_channel(x, sr, fake, xo)
            assert rep["added_500hz_below_crossover_db"] <= process.LEAK_LIMIT_DB


def test_frame_counts_do_not_depend_on_the_chunks(fake, monkeypatch):
    x = speechlike(22050, 7)
    orig = soxr.resample(x, 22050, RATE, quality="VHQ").astype(np.float64)
    model_in = fake.model_input(x, 22050)
    _, whole = process.added_band(orig, model_in, fake, 10000)
    monkeypatch.setattr(process, "CHUNK_S", 2.0)
    monkeypatch.setattr(process, "CONTEXT_S", 0.5)
    assert len(process._joins(len(orig))) > 2
    _, chunked = process.added_band(orig, model_in, fake, 10000)
    assert abs(chunked["frames"] - whole["frames"]) <= 0.005 * whole["frames"], (whole, chunked)
