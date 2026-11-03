import numpy as np
import pytest
import soxr

from conftest import noise_from, speechlike
from synchain_bwe import cutoff


def at48(x, sr):
    return soxr.resample(x, sr, 48000, quality="VHQ").astype(np.float64)


@pytest.mark.parametrize("sr,expected", [(16000, 7000), (22050, 10000), (24000, 11000), (32000, 15000)])
def test_resampled_noise(sr, expected):
    xo, info = cutoff.detect(at48(noise_from(sr, 4), sr))
    assert xo == expected, info


def test_speechlike_22k():
    xo, info = cutoff.detect(at48(speechlike(22050, 6), 22050))
    assert xo == 10000, info


def test_full_band_has_no_cutoff():
    xo, info = cutoff.detect(noise_from(48000, 4).astype(np.float64))
    assert xo is None and info["drop_db"] < cutoff.MIN_DROP_DB


def test_silence_does_not_crash():
    xo, _ = cutoff.detect(np.zeros(48000))
    assert xo is None


def as_16bit_44k(x, sr):
    """x made at sr, kept in a 44.1 kHz 16-bit file: the rounding fills the file's whole band with a noise floor."""
    y = soxr.resample(x, sr, 44100, quality="VHQ")
    return np.round(y * 32767) / 32767


def test_44k_16bit_file_holding_a_22k_voice():
    x = at48(as_16bit_44k(speechlike(22050, 6), 22050), 44100)
    xo, info = cutoff.detect(x, 44100)
    assert xo == 10000, info


def test_44k_16bit_full_band_has_no_cutoff():
    x = at48(as_16bit_44k(noise_from(44100, 4), 44100), 44100)
    xo, info = cutoff.detect(x, 44100)
    assert xo is None, info


@pytest.mark.parametrize("sr,expected", [(16000, 7000), (22050, 10000), (32000, 15000)])
def test_band_end_at_the_files_own_nyquist(sr, expected):
    xo, info = cutoff.detect(at48(noise_from(sr, 4), sr), sr)
    assert xo == expected, info


def test_nan_does_not_count_as_a_cutoff():
    x = noise_from(48000, 2).astype(np.float64)
    x[1000] = np.nan
    assert cutoff.detect(x)[0] is None
