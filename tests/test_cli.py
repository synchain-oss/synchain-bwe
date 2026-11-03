import json
import os

import numpy as np
import pytest
import soundfile as sf
import soxr

from conftest import FakeGuesser, noise_from, speechlike
from synchain_bwe import cli, model


def patch(monkeypatch):
    monkeypatch.setattr(model, "Guesser", lambda folder, threads=None: FakeGuesser())
    monkeypatch.setattr(model, "verify", lambda d: d)
    monkeypatch.setattr(model, "fetch", lambda d=None: "unused")


def test_process_and_refusals(tmp_path, monkeypatch):
    patch(monkeypatch)
    src = tmp_path / "in.wav"
    sf.write(src, np.stack([speechlike(22050, 3), speechlike(22050, 3, seed=1)], 1), 22050, subtype="PCM_16")
    ref = tmp_path / "ref.wav"
    sf.write(ref, speechlike(48000, 12, seed=2), 48000, subtype="FLOAT")
    out = tmp_path / "out.wav"
    assert cli.main(["process", str(src), "-o", str(out), "-r", str(ref), "--report", str(tmp_path / "r.json")]) == 0
    y, sr = sf.read(out)
    assert sr == 48000 and y.shape[1] == 2
    rep = json.loads((tmp_path / "r.json").read_text())[0]
    assert rep["crossover_hz"] == 10000 and len(rep["ceiling"]) == 6
    assert cli.main(["process", str(src), "-o", str(out)]) == 1  # exists
    assert cli.main(["process", str(src), "-o", str(src), "--force"]) == 1  # would overwrite the input


def test_full_band_input_is_skipped(tmp_path, monkeypatch):
    patch(monkeypatch)
    src = tmp_path / "full.wav"
    sf.write(src, noise_from(48000, 2), 48000)
    assert cli.main(["process", str(src), "-d", str(tmp_path / "o")]) == 0
    assert not (tmp_path / "o" / "full.wav").exists()


def test_ceiling_file(tmp_path, monkeypatch):
    patch(monkeypatch)
    ref = tmp_path / "ref.wav"
    sf.write(ref, speechlike(48000, 12), 48000, subtype="FLOAT")
    src = tmp_path / "in.wav"
    sf.write(src, speechlike(24000, 3), 24000)
    c = tmp_path / "c.json"
    assert cli.main(["ceiling", str(ref), "-o", str(c), "--like", str(src)]) == 0
    assert json.loads(c.read_text())["crossover_hz"] == 11000
    assert cli.main(["process", str(src), "-d", str(tmp_path / "o"), "-c", str(c), "--band-only"]) == 0
    assert (tmp_path / "o" / "in.band.wav").exists()
    other = tmp_path / "in22.wav"
    sf.write(other, speechlike(22050, 3), 22050)
    assert cli.main(["process", str(other), "-d", str(tmp_path / "o"), "-c", str(c)]) == 1  # the ceiling is for 11 kHz


def test_wildcards_and_name_clashes(tmp_path, monkeypatch):
    patch(monkeypatch)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    for d in ("a", "b"):
        sf.write(tmp_path / d / "take.wav", speechlike(22050, 2), 22050)
    sf.write(tmp_path / "a" / "other.wav", speechlike(22050, 2, seed=3), 22050)
    out = tmp_path / "o"
    assert cli.main(["process", str(tmp_path / "*" / "*.wav"), "-d", str(out)]) == 1  # a/take.wav and b/take.wav clash
    assert (out / "take.wav").exists() and (out / "other.wav").exists()
    assert cli.main(["process", str(tmp_path / "a" / "*.wav"), "-d", str(tmp_path / "a"), "--force"]) == 1  # onto the inputs


def test_nothing_protected_is_overwritten(tmp_path, monkeypatch):
    patch(monkeypatch)
    src = tmp_path / "in.wav"
    sf.write(src, speechlike(22050, 2), 22050)
    ref = tmp_path / "ref.wav"
    sf.write(ref, speechlike(48000, 12, seed=2), 48000, subtype="FLOAT")
    before = {p: p.read_bytes() for p in (src, ref)}
    for report in (src, ref):  # --report onto an input or the reference
        with pytest.raises(SystemExit):
            cli.main(["process", str(src), "-o", str(tmp_path / "o.wav"), "-r", str(ref), "--report", str(report), "--force"])
    assert cli.main(["process", str(src), "-o", str(ref), "-r", str(ref), "--force"]) == 1  # the output onto the reference
    link = tmp_path / "link.wav"
    os.link(src, link)
    assert cli.main(["process", str(src), "-o", str(link), "--force"]) == 1  # a hard link to the input
    with pytest.raises(SystemExit):
        cli.main(["ceiling", str(ref), "-o", str(ref), "--crossover", "10000", "--force"])
    assert {p: p.read_bytes() for p in (src, ref)} == before
    (tmp_path / "r.json").write_text("[]")
    with pytest.raises(SystemExit):  # an existing report needs --force
        cli.main(["process", str(src), "-o", str(tmp_path / "o.wav"), "--report", str(tmp_path / "r.json")])
    with pytest.raises(SystemExit):
        cli.main(["process", str(src), "-o", str(tmp_path / "o.mp3")])


def test_nan_input_fails_without_output(tmp_path, monkeypatch):
    patch(monkeypatch)
    x = speechlike(22050, 2)
    x[10] = np.nan
    src = tmp_path / "nan.wav"
    sf.write(src, x, 22050, subtype="FLOAT")
    assert cli.main(["process", str(src), "-o", str(tmp_path / "o.wav"), "--report", str(tmp_path / "r.json")]) == 1
    assert not (tmp_path / "o.wav").exists()
    assert "NaN" in json.loads((tmp_path / "r.json").read_text())[0]["error"]


def test_44k_16bit_input(tmp_path, monkeypatch):
    patch(monkeypatch)
    y = soxr.resample(speechlike(22050, 3), 22050, 44100, quality="VHQ")
    src = tmp_path / "in44.wav"
    sf.write(src, y, 44100, subtype="PCM_16")
    assert cli.main(["process", str(src), "-o", str(tmp_path / "o.wav"), "--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())[0]["crossover_hz"] == 10000


def test_report_and_ceiling_names(tmp_path, monkeypatch):
    patch(monkeypatch)
    src = tmp_path / "in.wav"
    sf.write(src, speechlike(22050, 2), 22050)
    for report in ("o.wav.", "o.wav ", "r.txt"):  # Windows drops a trailing dot or space: "o.wav." is o.wav
        with pytest.raises(SystemExit):
            cli.main(["process", str(src), "-o", str(tmp_path / "o.wav"), "--report", str(tmp_path / report)])
    ref = tmp_path / "ref.wav"
    sf.write(ref, speechlike(48000, 12, seed=2), 48000, subtype="FLOAT")
    for xo in ("30000", "2000"):
        with pytest.raises(SystemExit):
            cli.main(["ceiling", str(ref), "-o", str(tmp_path / "c.json"), "--crossover", xo])
    with pytest.raises(SystemExit):
        cli.main(["ceiling", str(ref), "-o", str(tmp_path / "c.txt"), "--crossover", "10000"])
    assert not (tmp_path / "c.json").exists()


def test_detect_goes_on_after_a_bad_file(tmp_path, capsys):
    x = speechlike(22050, 2)
    x[3] = np.nan
    sf.write(tmp_path / "a_nan.wav", x, 22050, subtype="FLOAT")
    (tmp_path / "b_text.wav").write_text("not audio")
    sf.write(tmp_path / "c_ok.wav", speechlike(22050, 2), 22050)
    assert cli.main(["detect", str(tmp_path / "*.wav")]) == 1
    assert "crossover 10000" in capsys.readouterr().out
