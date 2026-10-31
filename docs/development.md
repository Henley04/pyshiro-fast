# Development

## Setup

```bash
git clone https://github.com/henley04/pyshiro-fast
cd pyshiro-fast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Repository layout

```
pyshiro/
  align.py       alignment engine: DP kernel, Hirschberg, CLI
  model.py       .hsmm reader/writer, GMM/Duration dataclasses
  features.py    MFCC / Δ / ΔΔ front end (SHIRO-compatible)
  labels.py      .lab, TextGrid, Audacity readers and writers
  phonemes.py    kana-to-phoneme conversion
  train.py       EM training, CLI
  untie.py       monophone → triphone expansion
  bundled.py     access to the bundled models
  data/          kana2phonemes.table, models/*.hsmm
tests/           pytest suite (38 tests)
bench/           speed and memory benchmark
docs/            Markdown sources, shared stylesheet, generated HTML
examples/        runnable examples (synthetic demo, file aligner)
scripts/         docs build, release checks
example/         upstream sample lyrics and labels
```

## Tests

```bash
pytest                      # all
pytest tests/test_viterbi_exact.py -v
```

`tests/test_viterbi_exact.py` is the important one. It compares the optimised kernel
against the original `_viterbi_dp` on 13 configurations — topology skips, `pskip` edges,
mixed transitions, duration-free mode, prefix-topology variants — asserting both the
segments and the log-likelihood match exactly. Any change to `_dp_forward`,
`_dp_forward_call` or `_hirschberg_backtrace` must keep it green.

Tests that need the bundled models locate them through `pyshiro.bundled_model_path()`, so
they work both from a source checkout and from an installed wheel.

## Benchmark

```bash
python bench/bench_align_speed.py --sweep 10,20,60,120,300
python bench/bench_align_speed.py --wav song.wav --lyric song.txt
```

`--mem-mb 0` forces the divide-and-conquer path, which is how the two traceback strategies
are compared. The sweep uses features sampled deterministically from the model's GMMs;
DP cost depends only on `(T, N)`, so the numbers transfer to real audio. The GTSinger
numbers in [Performance](performance.md#validation) were produced with
`bench/real_audio_check.py`, which takes a plain phone list and an optional reference
timing.

To compare against upstream, install it in a separate environment and run the same sweep
with `bench/bench_align_speed.py` from that checkout.

## Documentation site

The site is plain HTML generated from `docs/*.md`. No external toolchain:

```bash
python scripts/build_docs.py        # writes docs/*.html
```

The generator uses a small Markdown subset (headings, paragraphs, lists, tables, fenced
code, links, inline code, bold/italic) and a single stylesheet: white background, system
fonts, 720 px measure. Rebuild and commit the HTML whenever you change a `.md` file.

GitHub Pages serves the `docs/` directory of `main` directly — no Actions step is
required.

## Releasing

1. Bump `__version__` in `pyshiro/__init__.py`. This is the single source of truth;
   `pyproject.toml` reads it dynamically.
2. Update `CHANGELOG.md`.
3. Check that the tree is clean and tests pass:

   ```bash
   pytest
   python scripts/check_release.py
   ```

4. Build and inspect:

   ```bash
   python -m build
   twine check dist/*
   ```

5. Verify the wheel in a clean environment — this catches missing package data:

   ```bash
   python -m venv /tmp/verify && /tmp/verify/bin/pip install dist/*.whl
   /tmp/verify/bin/python -c "import pyshiro; print(pyshiro.load_bundled_model()[0].nstate)"
   ```

6. Tag and push:

   ```bash
   git tag -a v1.0.0 -m "v1.0.0"
   git push origin main --tags
   ```

7. Publish. The `publish.yml` workflow uploads to PyPI via trusted publishing when a
   `v*` tag is pushed. It requires a PyPI trusted publisher configured for this
   repository and a `pypi` environment in GitHub settings. Manual fallback:

   ```bash
   twine upload dist/*
   ```

## Code style

- No formatter is enforced; match the surrounding code.
- Comments and docstrings may be Japanese in inherited modules — leave them. New public
  API docstrings and all user-facing strings are English.
- The DP kernel is performance-critical: measure before and after any change, and keep
  the equivalence tests green.
