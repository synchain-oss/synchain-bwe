"""Command line. Run `synchain-bwe -h` or `python -m synchain_bwe -h`."""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from . import __version__
from . import ceiling as ceil
from . import cutoff as cut
from .dsp import RATE
from .process import LOW_XO, NothingToDo, check_crossover, process_channel, resolve_crossover


def _read(path):
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if len(x) == 0:
        raise ValueError(f"{path} is empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{path} has NaN or infinite samples")
    return x, sr


def _mono48(path):
    """(the file as mono at 48 kHz, the file's own sample rate)."""
    x, sr = _read(path)
    m = x.mean(axis=1).astype(np.float32)
    return soxr.resample(m, sr, RATE, quality="VHQ").astype(np.float64), sr


def _same(a, b):
    """Whether two paths are, or would be, the same file: the same resolved name, or the same file on disk (hard links,
    case-insensitive file systems)."""
    a, b = Path(a), Path(b)
    if a.resolve() == b.resolve():
        return True
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _expand(items):
    """Inputs with wildcards expanded here, because Windows shells leave *.wav as it is."""
    out = []
    for it in items:
        if any(c in it for c in "*?[") and not Path(it).exists():
            hits = sorted(glob.glob(it))
            if not hits:
                sys.exit(f"nothing matches {it}")
            out += hits
        else:
            out.append(it)
    return out


def _out_path(inp, args, multi):
    if args.output and not multi:
        return Path(args.output)
    suffix = ".band.wav" if args.band_only else ".wav"
    return Path(args.out_dir) / (Path(inp).stem + suffix)


def cmd_process(args):
    args.inputs = _expand(args.inputs)
    if len(args.inputs) > 1 and not args.out_dir:
        sys.exit("several inputs: give --out-dir")
    if not args.output and not args.out_dir:
        sys.exit("give --output (one input) or --out-dir")
    if args.reference and args.ceiling:
        sys.exit("give --reference or --ceiling, not both")
    if args.mode == "raw" and (args.reference or args.ceiling):
        sys.exit("--mode raw uses the model's guess as it is: no ceiling")
    if args.output and len(args.inputs) == 1 and Path(args.output).suffix.lower() != ".wav":
        sys.exit(f"{args.output}: the output is a WAV file; give it a .wav name")
    if args.report and Path(args.report).suffix.lower() != ".json":
        sys.exit(f"{args.report}: the report is a JSON file; give it a .json name")
    # Files this run never writes: the inputs, the reference and the ceiling file.
    protected = list(args.inputs) + [p for p in (args.reference, args.ceiling) if p]
    outputs = [_out_path(inp, args, len(args.inputs) > 1) for inp in args.inputs]
    if args.report:
        if any(_same(args.report, p) for p in protected + outputs):
            sys.exit(f"--report {args.report} would overwrite an input, the reference, the ceiling file or an output")
        if Path(args.report).exists() and not args.force:
            sys.exit(f"{args.report} exists (use --force to overwrite)")
    from .model import Guesser, fetch, verify

    folder = verify(args.model_dir) if args.model_dir else fetch()
    guesser = Guesser(folder, threads=args.threads)
    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    try:
        ref = _mono48(args.reference) if args.reference else None
        saved = ceil.load(args.ceiling) if args.ceiling else None
    except (ValueError, OSError, sf.LibsndfileError) as e:
        sys.exit(str(e))
    ceilings = {}
    reports, failed, planned = [], 0, []
    try:
        for inp, out in zip(args.inputs, outputs):
            rep = {"input": str(inp), "output": str(out)}
            try:
                if any(_same(out, p) for p in protected):
                    raise ValueError("the output would overwrite an input, the reference or the ceiling file")
                if any(_same(out, p) for p in planned):
                    raise ValueError(f"another input in this run also writes {out} (same file name)")
                planned.append(out)
                if out.exists() and not args.force:
                    raise ValueError(f"{out} exists (use --force to overwrite)")
                _process_one(inp, out, args, guesser, ref, saved, ceilings, rep)
            except NothingToDo as e:
                rep["skipped"] = str(e)
                print(f"{inp}: skipped: {e}", file=sys.stderr)
            except (ValueError, RuntimeError, OSError, sf.LibsndfileError) as e:
                rep["error"] = str(e)
                failed += 1
                print(f"{inp}: {e}", file=sys.stderr)
            reports.append(rep)
    finally:
        if args.report:
            Path(args.report).write_text(json.dumps(reports, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if failed else 0


def _process_one(inp, out, args, guesser, ref, saved, ceilings, rep):
    x, sr = _read(inp)
    first48 = soxr.resample(x.mean(axis=1), sr, RATE, quality="VHQ").astype(np.float64)
    xo, how = resolve_crossover(first48, args.crossover, sr)
    check_crossover(xo)
    if xo < LOW_XO:
        print(f"{inp}: the crossover is at {xo} Hz; synchain-bwe has not been tested on inputs that stop below {LOW_XO} Hz", file=sys.stderr)
    c = None
    if ref is not None:
        if xo not in ceilings:
            ceilings[xo] = ceil.from_reference(ref[0], xo, ref[1])
        c = ceilings[xo]
    elif saved is not None:
        if saved["crossover_hz"] != xo:
            raise ValueError(f"the ceiling file is for a crossover at {saved['crossover_hz']} Hz, this input's is at {xo} Hz "
                             f"(make one with: synchain-bwe ceiling REF --crossover {xo} -o CEILING.json)")
        c = saved
    ys, chans = [], []
    for ch in range(x.shape[1]):
        y, r = process_channel(x[:, ch], sr, guesser, xo, c["ceiling"] if c else None, args.mode, args.band_only)
        ys.append(y)
        chans.append(r)
    y = np.stack(ys, axis=1)
    peak = float(np.abs(y).max())
    subtype = "FLOAT" if args.float else "PCM_24"
    if not np.isfinite(peak):
        raise RuntimeError("the output has NaN or infinite samples; this is a bug")
    if peak >= 1.0 and subtype != "FLOAT":
        raise ValueError(f"the output would clip (peak {peak:.3f}); use --float, or lower the input's level")
    sf.write(str(out), y if y.shape[1] > 1 else y[:, 0], RATE, subtype=subtype, format="WAV")
    rep.update({"input_rate": sr, "channels": x.shape[1], "seconds": round(len(x) / sr, 3), "crossover_hz": xo,
                "crossover_found": how, "mode": args.mode, "band_only": args.band_only, "peak": round(peak, 4),
                "ceiling": c["ceiling"] if c else None, "channel_reports": chans})
    print(f"{inp} -> {out}: crossover {xo} Hz, step {chans[0]['step_db_before']} -> {chans[0]['step_db_after']} dB")


def cmd_ceiling(args):
    if (args.crossover is None) == (args.like is None):
        sys.exit("give --crossover or --like")
    if Path(args.output).suffix.lower() != ".json":
        sys.exit(f"{args.output}: the ceiling file is a JSON file; give it a .json name")
    if any(_same(args.output, p) for p in (args.reference, args.like) if p):
        sys.exit("the ceiling file would overwrite the reference or the --like input")
    if Path(args.output).exists() and not args.force:
        sys.exit(f"{args.output} exists (use --force to overwrite)")
    try:
        if args.like:
            xo, _ = cut.detect(*_mono48(args.like))
            if xo is None:
                sys.exit(f"{args.like}: no clear cutoff")
        else:
            xo = args.crossover
        check_crossover(xo)
        x, sr = _mono48(args.reference)
        c = ceil.from_reference(x, xo, sr)
    except NothingToDo as e:
        sys.exit(f"crossover {xo} Hz: {e}")
    except (ValueError, OSError, sf.LibsndfileError) as e:
        sys.exit(str(e))
    ceil.save(c, args.output)
    print(f"ceiling for a crossover at {xo} Hz: " + ", ".join(f"{a / 1000:g} kHz {b:+.1f} dB" for a, b in c["ceiling"]))
    return 0


def cmd_detect(args):
    failed = 0
    for inp in _expand(args.inputs):
        try:
            xo, info = cut.detect(*_mono48(inp))
        except (ValueError, OSError, sf.LibsndfileError) as e:
            print(f"{inp}: {e}", file=sys.stderr)
            failed += 1
            continue
        if xo is None:
            print(f"{inp}: no clear cutoff (steepest fall {info['drop_db']} dB at {info['boundary_hz']} Hz)")
        else:
            print(f"{inp}: falls 6 dB by {info['minus_6db_hz']} Hz; crossover {xo} to {xo + 500} Hz")
    return 1 if failed else 0


def cmd_fetch(args):
    from .model import fetch

    print(fetch(args.model_dir))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="synchain-bwe", description="Adds the missing high band to band-limited voices, "
                                 "such as AI voices made at 16, 22.05 or 24 kHz, with LavaSR's guess re-levelled to the voice's own level.")
    ap.add_argument("--version", action="version", version=f"synchain-bwe {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("process", help="add the high band to one or more files")
    p.add_argument("inputs", nargs="+")
    p.add_argument("-o", "--output", help="output .wav file (one input)")
    p.add_argument("-d", "--out-dir", help="output folder (any number of inputs; names kept)")
    p.add_argument("-r", "--reference", help="a full-band recording of a real voice (44.1 or 48 kHz; at least 4 s of speech, 10 s or more is best) to take the ceiling from")
    p.add_argument("-c", "--ceiling", help="a ceiling file made with `synchain-bwe ceiling`, instead of --reference")
    p.add_argument("--crossover", type=int, help="where the crossover starts, in Hz (default: found from each input)")
    p.add_argument("--mode", choices=("matched", "raw"), default="matched", help="matched (default): re-levelled; raw: the model's guess as it is")
    p.add_argument("--band-only", action="store_true", help="write only the added band, to lay under the original at 48 kHz")
    p.add_argument("--float", action="store_true", help="write 32-bit float instead of 24-bit")
    p.add_argument("--model-dir", help="a folder with LavaSR's enhancer_v2 files (checked by SHA-256); default: download once and cache")
    p.add_argument("--threads", type=int, help="CPU threads for the model")
    p.add_argument("--report", help="write a JSON report to this .json file (it includes the ceiling numbers)")
    p.add_argument("--force", action="store_true", help="overwrite existing outputs and report (inputs, reference and ceiling file never)")
    p.set_defaults(fn=cmd_process)

    p = sub.add_parser("ceiling", help="measure a ceiling from a reference recording and save it")
    p.add_argument("reference")
    p.add_argument("-o", "--output", required=True, help="the ceiling file to write (.json)")
    p.add_argument("--crossover", type=int, help="crossover start in Hz")
    p.add_argument("--like", help="or: take the crossover from this input file")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_ceiling)

    p = sub.add_parser("detect", help="show where each input stops and where the crossover would go")
    p.add_argument("inputs", nargs="+")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("fetch", help="download and check the model files")
    p.add_argument("--model-dir", help="put them in this folder instead of the Hugging Face cache")
    p.set_defaults(fn=cmd_fetch)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
