# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-10-31

First release of `pyshiro-fast`, a fork of
[pyshiro](https://github.com/wavtechyukky/pyshiro) 0.1.0 with a rewritten alignment
kernel.

### Added

- **Fast Viterbi decoder** (`pyshiro/align.py`): exact pruning of the duration axis,
  `O(1)` sliding-window argmax in the duration-free pass, packed 32-bit backpointers
  with a `K`-row score ring buffer, and Hirschberg divide-and-conquer traceback above a
  configurable memory budget. Long-audio alignment is 100–700× faster; 10 minutes of
  audio goes from ~69 h (extrapolated) to 5.7 minutes.
- **`backtrace_mem_mb`** keyword on `forced_align` and `forced_align_2pass`
  (default 1024 MiB). Set `0` to force the low-memory path.
- **Bundled models inside the wheel** (`pyshiro/data/models/`), with
  `pyshiro.load_bundled_model()` and `pyshiro.bundled_model_path()`.
- **`pyshiro_fast` alias package**, so the distribution name can be imported directly.
- **`tests/test_viterbi_exact.py`**: 13 equivalence tests comparing the new kernel with
  the original `_viterbi_dp` on topology skips, `pskip` edges, mixed transitions,
  duration-free mode and prefix-topology variants. Segments and log-likelihood must match
  exactly.
- **`bench/bench_align_speed.py`**: scaling sweep and real-audio benchmark.
- **`bench/real_audio_check.py`**: align one recording from a plain phone list and print
  sanity statistics (obstruent durations, optional deviation from a reference timing).
- **Documentation site** under `docs/` (static HTML, GitHub Pages ready).

### Validated

- **Exactness.** Segments and log-likelihood are bit-identical to upstream on
  deterministic GMM-sampled features at 20/60/120 s, on the two traceback strategies at
  300 s, and on four real GTSinger Japanese clips (18–32 s, two singers, four
  techniques).
- **Usability on real singing.** Obstruent durations stay at 60–95 ms median (max 230 ms)
  across the four clips; word boundaries deviate from the corpus annotation by 40–76 ms
  median, the order of the annotation's own 30 ms quantisation. One clip is excluded from
  that figure because its annotation is degenerate — see
  [docs/performance.md](docs/performance.md#validation).

### Changed

- Package renamed to `pyshiro-fast`; the import name stays `pyshiro`.
- `__version__` is 1.0.0 and is the single source of truth for the package metadata.
- CLI help, progress output and error messages are English.
- `pyshiro-align` no longer requires `--model` / `--phonemap`; it defaults to the bundled
  `jp-v2` model.
- `python -m pyshiro.align` now runs the same `main()` as the console script instead of a
  separate demo with hard-coded paths.
- `pyshiro/__init__.py` also exports the label readers, `write_audacity` and
  `segments_to_phoneme_intervals`, which were previously only importable from
  `pyshiro.labels`.

### Fixed

- `tests/test_align.py` asserted the pre-0.1.0 return type of `forced_align` (a bare list);
  it now expects the `(segments, loglik)` tuple. The test failed on upstream 0.1.0 as
  well.

### Compatibility

- Alignment results are bit-identical to upstream 0.1.0: same segments, same `loglik`.
- `.hsmm`, `phonemap.json`, `.lab`, TextGrid and Audacity formats are unchanged.
- Public function signatures are unchanged; only optional keyword arguments were added.

[1.0.0]: https://github.com/henley04/pyshiro-fast/releases/tag/v1.0.0
