# Contributing

Bug reports and pull requests are welcome.

## Set up

The tests use a stand-in for the model, so they need neither torch nor the model download.

```sh
python -m pip install numpy soundfile soxr pyyaml pytest ruff
python -m pytest -q
ruff check .
```

## What must stay out of this repository

No audio, model files, ceiling files or reports, and no numbers measured from a real voice, in code, tests, docs or
comments. Those describe a real person's voice, and git history is public for good. Tests use synthetic signals.
`.gitignore` is an allow-list and CI checks the whole history, so a new kind of file also needs a change to both.

## Pull requests

- Branch from `main`, and name the branch `feat/...` or `feature/...`. PRs from forks can use any name except `main`.
- Sign off every commit with `git commit -s`. That adds a `Signed-off-by` line, which certifies the
  [Developer Certificate of Origin](https://developercertificate.org/): you have the right to submit the change under
  this project's MIT licence. To fix commits you forgot, `git rebase --signoff main`.
- `bwe-gate` and `branch-gate` must pass before merging. AI reviewers comment on PRs; a maintainer decides.
