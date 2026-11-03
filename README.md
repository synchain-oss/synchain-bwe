# synchain-bwe

Adds the missing high band to band-limited voices, such as AI voices generated at 16, 22.05 or 24 kHz, on your own
computer. A neural model ([LavaSR](https://github.com/ysharma3501/LavaSR)) guesses what lies above the cutoff;
synchain-bwe keeps only the shape of that guess and takes its level from the voice itself, frame by frame, optionally
capped at how bright a real voice gets.

> **Status:** early (0.1.0). So far it has been tested on one voice: 77 lines of Chinese speech from IndexTTS 2.0 at
> 22.05 kHz. Listen to the result before you use it.

[中文说明](README.zh-CN.md)

## How it works

Below the crossover the output is the input, resampled to 48 kHz and otherwise unchanged. Above it, a band is added:

1. **Guess.** LavaSR's bandwidth-extension model hears the input through 16 kHz, as LavaSR's own pipeline feeds it, and
   guesses the full band up to 24 kHz.
2. **Level.** In every frame (10.7 ms long, 2.7 ms apart) the guess is scaled so that its edge band (the 1.5 kHz just
   below the crossover, which the input really has) matches the input's own level there. So the added band follows the
   voice: as bright as the voice is at that moment, and silent where the voice is silent. At most +24 dB.
3. **Roll-off.** From 16 to 20 kHz the band fades to nothing, which removes the model's steady hiss up there.
4. **Ceiling** (with `--reference`). From a full-band recording of a real voice: above the crossover, each stretch of the
   spectrum is held below the level the reference exceeds in only 10% of its speech frames, relative to the same edge
   band. The frames are put back together, measured and capped again, four times over.
5. **Crossover.** The added band replaces what the input has above the crossover start, through a 500 Hz smoothstep
   crossover. The crossover start sits a little below where the input starts to fall (its cutoff); it is found from
   each file (`detect` shows both) or given with `--crossover`. The band fades in and out over 20 ms at the ends of
   each file.

Why not use the model's output as it is? On the voice it was tested on, the guess came out about 15 dB quieter than the
voice's real content just below the cutoff, and above 16 kHz it was mostly a steady hiss that did not follow the voice,
pauses included. `--mode raw` writes that version, for comparison.

## Measured on one voice

77 lines of an IndexTTS 2.0 voice at 22.05 kHz (full to about 10.3 kHz, nothing above 11 kHz). The reference was a 19 s
48 kHz recording of the person the AI voice was made from, used with their consent.

| | Input | `--mode raw` | matched, with reference |
|---|---|---|---|
| Step at the cutoff (9.5-10 kHz against 10.5-11 kHz) | about 22 dB | 17.1 dB | 5.0 dB (3.2 to 6.9 per line) |
| Added band in pauses | none | steady hiss | about 26 dB below raw |

Compared with the real speaker, each frame relative to its own 8.5 to 10 kHz: in sibilants the added band was within
3 dB of the speaker, the brightest tenth of sibilants 2 to 8 dB duller; in other speech (mostly vowels) it was 3.5 to
5.5 dB brighter at 12 to 18 kHz. Averaged over whole lines, 10.5 to 16 kHz came out 3 to 4 dB darker than the speaker
and 16 to 20 kHz level with them.

Known limits:

- About 1% of speech frames stay more than 3 dB over the ceiling after the re-capping, at most about 13 dB. They sit in
  the frame just before a sharp sibilant onset: the model's band starts about 2 ms early (a median of 1.8 ms on test
  bursts), quietly, 15 to 30 dB under the sibilant that follows. For comparison, the reference recording is more than
  3 dB over its own ceiling in about 11% of its speech frames.
- Without a reference there is no ceiling, only the +24 dB limit and the roll-off.
- synchain-bwe has not been tested on inputs that stop below 7 kHz (LavaSR itself says it takes inputs from an 8 kHz
  sample rate up).

## Install

Python 3.10 or newer and git.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS / Linux: source .venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # optional: the smaller CPU build
pip install "synchain-bwe @ git+https://github.com/synchain-oss/synchain-bwe"
```

The first run downloads two files (about 56 MB) from Hugging Face, the `enhancer_v2` part of
[YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR) at a fixed revision, and checks both by SHA-256 before
loading them. `synchain-bwe fetch` does this ahead of time; `--model-dir` uses a copy you already have.

## Use

```sh
synchain-bwe detect voice.wav                                   # where the input stops, where the crossover goes
synchain-bwe process voice.wav -o voice_48k.wav -r real_voice_48k.wav
synchain-bwe process "takes/*.wav" -d out/ -r real_voice_48k.wav
synchain-bwe ceiling real_voice_48k.wav --like voice.wav -o voice_ceiling.json
synchain-bwe process "takes/*.wav" -d out/ -c voice_ceiling.json  # the saved ceiling instead of the recording
synchain-bwe process voice.wav -o band.wav --band-only          # only the band, to lay under the original at 48 kHz
synchain-bwe process voice.wav -o raw.wav --mode raw            # the model's guess as it comes, for comparison
```

| Option | |
|---|---|
| `-o`, `-d` | output file (one input) or output folder (any number; names kept, `.band.wav` with `--band-only`) |
| `-r`, `-c` | reference recording, or a ceiling file made with `synchain-bwe ceiling` |
| `--crossover HZ` | where the crossover starts; default: found per file |
| `--mode raw` | the model's guess without the re-levelling |
| `--band-only` | write only the added band |
| `--float` | 32-bit float output instead of 24-bit |
| `--model-dir DIR` | use model files from this folder (still checked by SHA-256) |
| `--threads N` | CPU threads for the model |
| `--report FILE.json` | JSON report per file: crossover, step before and after, frames at the +24 dB limit, frames capped, and the ceiling used |
| `--force` | overwrite existing outputs and report (the inputs, the reference and the ceiling file are never overwritten) |

Output is always 48 kHz WAV with the input's channels; each channel is done on its own. Files with no clear cutoff are
skipped and reported. Long files are done in 30 s chunks; the joins match a single pass over the file far below
24-bit resolution (measured: -175 dB with the model on the test voice, about -150 dB in the tests). On one desktop CPU (an AMD Ryzen 9 9950X) using 8 threads it ran about 15 times faster than real time.

### The reference recording

Use a clean recording of a real voice close to the AI voice, best the person it was made from, with their consent:
44.1 or 48 kHz with its full band (nothing cut below 20 kHz), 10 to 30 s of speech (at least 4 s), no music or noise, no de-esser or
EQ. A reference that stops early, or has almost nothing above the crossover, is refused. The ceiling is taken relative to the band just
below the crossover, so a ceiling file only fits inputs with the same crossover (`--like` takes it from an input).

## Privacy

Everything runs on your computer. The only network access is the model download from Hugging Face, once. No audio is
sent anywhere. A ceiling file is a handful of numbers describing the reference voice; treat it as you would treat that.
The `--report` file holds the same numbers.

## Credits and licences

- synchain-bwe: MIT, see [LICENSE](LICENSE).
- [LavaSR](https://github.com/ysharma3501/LavaSR) by Yatharth Sharma provides the model and its weights
  ([YatharthS/LavaSR](https://huggingface.co/YatharthS/LavaSR)). Its LICENSE file and model card say Apache-2.0; its
  package metadata says MIT. synchain-bwe does not include or install LavaSR's code: it downloads the weights and runs
  them with its own loader, including LavaSR's one change to the Vocos head (the magnitude clipped at 1e3).
- [Vocos](https://github.com/charactr-platform/vocos) (MIT, Charactr Inc.), in the
  [langtech-bsc fork](https://github.com/langtech-bsc/vocos) LavaSR uses, at a fixed commit, is installed as a dependency.

synchain-bwe is not affiliated with or endorsed by the authors of LavaSR or Vocos.
