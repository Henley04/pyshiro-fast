# pyshiro-fast

**Fast HSMM forced alignment for Japanese singing voice.**
A drop-in fork of [pyshiro](https://github.com/wavtechyukky/pyshiro) that aligns long
recordings **100–700× faster** while producing the *same* alignment, frame for frame.

[![PyPI](https://img.shields.io/pypi/v/pyshiro-fast.svg)](https://pypi.org/project/pyshiro-fast/)
[![Python](https://img.shields.io/pypi/pyversions/pyshiro-fast.svg)](https://pypi.org/project/pyshiro-fast/)
[![CI](https://github.com/henley04/pyshiro-fast/actions/workflows/ci.yml/badge.svg)](https://github.com/henley04/pyshiro-fast/actions/workflows/ci.yml)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](https://github.com/henley04/pyshiro-fast/blob/main/LICENSE)

[Documentation](https://henley04.github.io/pyshiro-fast/) ·
[Quick start](https://henley04.github.io/pyshiro-fast/quickstart.html) ·
[Performance](https://henley04.github.io/pyshiro-fast/performance.html) ·
[API](https://henley04.github.io/pyshiro-fast/api.html)

---

## Why this fork

pyshiro aligns a song by running Viterbi over an HSMM: three states per phoneme,
durations modelled explicitly. That is accurate, but the duration-free first pass
enumerates every possible state duration, so its cost grows as `O(N·T²)` — cubic in
audio length. A three-minute song that takes seconds at 20 s takes hours at 5 minutes.

`pyshiro-fast` rewrites the decoder. Nothing else changes: same models, same phonemap,
same `.lab` / TextGrid output, same Python API, and — verified by regression tests —
**bit-identical results**, including the log-likelihood.

| Audio length | Frames `T` | States `N` | Upstream | pyshiro-fast | Speedup | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|
| 20 s | 4 000 | 1 005 | 11.09 s | **0.65 s** | 17× | 737 → 192 MB |
| 60 s | 12 000 | 3 006 | 251.4 s | **2.50 s** | 101× | 737 → 324 MB |
| 120 s | 24 000 | 6 006 | 1 986.5 s | **6.79 s** | 293× | 2 401 → 742 MB |
| 300 s | 60 000 | 15 006 | ≈ 8.7 h † | **91.7 s** | ≈ 341× | ≥ 14 GB ‡ | 255 MB |
| 600 s | 120 000 | 30 006 | ≈ 69 h † | **340.2 s** | ≈ 730× | ≥ 57 GB ‡ | 332 MB |

† Upstream values for 300 s and 600 s are extrapolated from the three measured points
(`t ≈ 1.155e-3·s³ + 1.85`, fit error < 0.6 %); upstream is genuinely `O(s³)`.
‡ Computed from the three `(N, T+1)` matrices upstream allocates (16 B per cell).
Rows 20/60/120 s are measured on one machine, same input, same JIT settings.

End-to-end on a real 40 s WAV through the full CLI (feature extraction → lyrics →
alignment → `.lab`): **22.7 s → 1.9 s**, and both versions write identical frame
boundaries.

## How it is faster

All four changes are in `pyshiro/align.py`. The original kernel is kept as
`_viterbi_dp` and is used as the reference implementation in tests.

1. **Exact pruning of the duration axis.** Instead of scanning `d` from `floor` to
   `min(ceil, t)`, the range is intersected with the time window the predecessor
   state can actually reach. Cells that are provably unreachable are skipped.
2. **`O(1)` argmax in the duration-free pass.** Without a duration prior,
   `score[n,t] = cum[t] + max_p(score[src,p] − cum[p])` is a sliding-window maximum,
   updated incrementally: `O(N·T²) → O(N·T)`. Values are still evaluated as
   `score[src,p] + (cum[t] − cum[p])` in the original order, so no rounding changes.
3. **Compact backpointers.** `d` and the predecessor offset are packed into one
   `int32` per cell (4 B instead of 16 B), and only the last `K` score rows are kept
   in a ring buffer.
4. **Hirschberg divide-and-conquer.** When the backpointer matrix would exceed
   `backtrace_mem_mb` (default 1024 MiB), the state axis is split at its midpoint,
   forward and reverse DP meet there, and the traceback recurses on both halves.
   Memory becomes `O(K·T)` instead of `O(N·T)`; output is unchanged.

The reverse pass reuses the forward kernel by reversing both the state order and the
time axis, so there is one DP implementation, not two.

## Install

```bash
pip install pyshiro-fast
```

Requires Python 3.10+, `numpy`, `scipy`, `soundfile`, `numba`, `msgpack`.
The wheel ships two pre-trained Japanese models (~650 KB), so nothing else needs to be
downloaded.

```python
import pyshiro                      # import name is unchanged
print(pyshiro.__version__)          # 1.0.0
```

`import pyshiro_fast` also works and exposes the same objects, if you prefer to import
the distribution name.

## Quick start

```python
import pyshiro

# 1. Load a bundled model (jp-v2, or "jp-v1" for the breath-aware model)
model, phonemap = pyshiro.load_bundled_model()

# 2. Features: 16 kHz mono WAV -> 3 streams (MFCC, Δ, ΔΔ)
streams = pyshiro.extract_mfcc_from_file("song.wav")

# 3. Lyrics: hiragana, one phrase per line -> phoneme sequence
phonemes = pyshiro.convert_lyric_file("song.txt")

# 4. Align (2-pass: HMM bootstrap, then HSMM refinement)
T = streams[0].shape[0]
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
segments, loglik = pyshiro.forced_align_2pass(
    model, streams, state_seq, nodur_phonemes={"pau", "br"})
print(f"{loglik / T:.3f} nats/frame")

# 5. Write labels
intervals = pyshiro.segments_to_phoneme_intervals(phonemes, segments)
pyshiro.write_lab(intervals, "song.lab")            # HTK 100 ns, ENUNU/NNSVS compatible
```

### CLI

```bash
# bundled model, .lab output next to the WAV
pyshiro-align song.wav song.txt

# explicit model and TextGrid output
pyshiro-align song.wav song.txt \
    --model my_model.hsmm --phonemap my_model_phonemap.json \
    --format textgrid --out song.TextGrid
```

`pyshiro-align --help` for all options. Training has its own entry point,
`pyshiro-train`, documented in [https://henley04.github.io/pyshiro-fast/training.html](https://henley04.github.io/pyshiro-fast/training.html).

## What stays the same

- Public API: `forced_align` still returns `(segments, loglik)`; all upstream
  functions keep their signatures. `forced_align` gained two optional keyword
  arguments (`backtrace_mem_mb`, `hmm_cap`), both with defaults matching upstream
  behaviour.
- File formats: `.hsmm`, `phonemap.json`, HTK `.lab`, TextGrid, Audacity labels.
- Numbers: identical segments and identical `loglik` bit pattern. The test suite
  compares the new kernel against `_viterbi_dp` on 13 configurations, including
  topology skips, `pskip` edges, mixed transitions and duration-free mode.

## Bundled models

| Name | File | Notes |
|---|---|---|
| `jp-v2` (default) | `pyshiro-jp-v2.hsmm` | `--nmix 4`; mean boundary error 2.05 frames on held-out singers |
| `jp-v1` | `pyshiro-jp-v1.hsmm` | 19.4 h multi-database; better on `br` (breath) |

```python
model, phonemap = pyshiro.load_bundled_model("jp-v1")
print(pyshiro.bundled_model_path("jp-v1"))   # (hsmm, phonemap) as Path objects
```

Both models come from the upstream project and are redistributed under GPL-3.0.
See [https://henley04.github.io/pyshiro-fast/models.html](https://henley04.github.io/pyshiro-fast/models.html) for training data credits.

## Documentation

| Page | Contents |
|---|---|
| [Installation](https://henley04.github.io/pyshiro-fast/installation.html) | pip, source install, troubleshooting |
| [Quick start](https://henley04.github.io/pyshiro-fast/quickstart.html) | end-to-end walkthrough, lyrics format, label output |
| [CLI reference](https://henley04.github.io/pyshiro-fast/cli.html) | `pyshiro-align`, `pyshiro-train`, `pyshiro.untie` |
| [Python API](https://henley04.github.io/pyshiro-fast/api.html) | every public function, with signatures and examples |
| [Performance](https://henley04.github.io/pyshiro-fast/performance.html) | benchmark method, scaling curves, memory, `backtrace_mem_mb` |
| [Migrating from pyshiro](https://henley04.github.io/pyshiro-fast/migration.html) | what changed, what is compatible, how to switch |
| [Training](https://henley04.github.io/pyshiro-fast/training.html) | corpus layout, HMM bootstrap, DAEM, `--nmix` |
| [Development](https://henley04.github.io/pyshiro-fast/development.html) | tests, benchmarks, release process |
| [FAQ](https://henley04.github.io/pyshiro-fast/faq.html) | common failure modes and fixes |

## Requirements

- Python 3.10 – 3.13
- Linux, macOS or Windows (numba-supported platforms)
- No GPU

## License and credits

GPL-3.0-only, inherited from the upstream project. See [LICENSE](https://github.com/henley04/pyshiro-fast/blob/main/LICENSE) and
[NOTICE](https://github.com/henley04/pyshiro-fast/blob/main/NOTICE).

This project builds on:

- [pyshiro](https://github.com/wavtechyukky/pyshiro) by wavtechyukky — the Python
  implementation this fork is based on
- [SHIRO](https://github.com/Sleepwalking/SHIRO) by Sleepwalking — the original HSMM
  alignment toolkit
- [ENUNU](https://github.com/oatsu-gh/ENUNU) — the kana-to-phoneme conversion table
  originates from its conversion data

The alignment decoder, the test suite additions and the packaging are new in this fork.

## Citation

```bibtex
@software{pyshiro_fast,
  title   = {pyshiro-fast: fast HSMM forced alignment for Japanese singing voice},
  author  = {{pyshiro-fast contributors}},
  year    = {2026},
  url     = {https://github.com/henley04/pyshiro-fast},
  note    = {GPL-3.0; fork of pyshiro by wavtechyukky}
}
```

## Contributing

Bug reports and pull requests are welcome — see [CONTRIBUTING.md](https://github.com/henley04/pyshiro-fast/blob/main/CONTRIBUTING.md).
Performance changes must keep the `tests/test_viterbi_exact.py` suite green; that
suite is what guarantees the output is unchanged.
