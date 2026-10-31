# Migrating from pyshiro

`pyshiro-fast` is a drop-in replacement for the upstream [pyshiro](https://github.com/wavtechyukky/pyshiro).
Scripts keep working; only paths to the bundled models change.

## Switch

```bash
pip uninstall pyshiro
pip install pyshiro-fast
```

The import name is still `pyshiro`:

```python
import pyshiro
print(pyshiro.__version__)    # 1.0.0 (upstream: 0.1.0)
```

## What is compatible

| Item | Status |
|---|---|
| `import pyshiro` / `pyshiro.<submodule>` | unchanged |
| `.hsmm` models, `phonemap.json` | unchanged, both directions |
| `.lab`, TextGrid, Audacity labels | unchanged |
| `forced_align`, `forced_align_2pass` signatures | unchanged, two optional keywords added |
| `refine_pau_boundaries`, `realign_external_labels` | unchanged |
| CLI `pyshiro-align`, `pyshiro-train` | unchanged arguments; help text is now English |
| Alignment output | identical segments and identical `loglik` |

## What changed

### Models moved into the package

Upstream kept models in `checkpoint/`. They now live in `pyshiro/data/models/` so they
can ship inside the wheel.

```python
# before
model    = pyshiro.load_hsmm("checkpoint/pyshiro-jp-v2.hsmm")
phonemap = pyshiro.load_phonemap("checkpoint/pyshiro-jp-v2_phonemap.json")

# after — either use the helper …
model, phonemap = pyshiro.load_bundled_model()

# … or keep explicit paths
model    = pyshiro.load_hsmm("/path/to/pyshiro-jp-v2.hsmm")
```

If your code hard-codes `checkpoint/`, replace it with `pyshiro.bundled_model_path()`:

```python
hsmm, pm = pyshiro.bundled_model_path("jp-v2")
```

### CLI no longer requires `--model` / `--phonemap`

```bash
# before
pyshiro-align song.wav song.txt \
    --model checkpoint/pyshiro-jp-v2.hsmm \
    --phonemap checkpoint/pyshiro-jp-v2_phonemap.json

# after
pyshiro-align song.wav song.txt
```

Passing them explicitly still works and takes precedence.

### Two new keyword arguments

`forced_align` and `forced_align_2pass` accept:

| Argument | Default | Effect |
|---|---|---|
| `backtrace_mem_mb` | `1024` | memory budget for the backpointer matrix; above it, divide-and-conquer |
| `hmm_cap` | `None` | max frames per state in the duration-free pass (already existed upstream; now passable through `forced_align_2pass`) |

Defaults reproduce upstream behaviour, except that the memory guard is active. On inputs
where upstream would allocate more than 1 GiB of backpointers, the divide-and-conquer
path runs instead — same result, 2–2.5× slower, much less memory. Raise the budget if
you prefer speed:

```python
segments, loglik = pyshiro.forced_align_2pass(..., backtrace_mem_mb=4096)
```

### Reference kernel kept for testing

Upstream's `_viterbi_dp` is still present and still callable. `tests/test_viterbi_exact.py`
uses it to assert that the new kernel gives the same segments and log-likelihood on 13
configurations. Do not delete it; it is the correctness oracle.

### Package renamed

| | Upstream | This fork |
|---|---|---|
| PyPI distribution | — (not published) | `pyshiro-fast` |
| Import name | `pyshiro` | `pyshiro` (alias `pyshiro_fast`) |
| Version | 0.1.0 | 1.0.0 |
| Console scripts | `pyshiro-align`, `pyshiro-train` | same |

The version jump reflects the API and format stability plus the decoder rewrite, not a
breaking change.

## Behavioural differences to be aware of

1. **Help and log output are English.** Argument names are unchanged; only strings.
2. **Bundled models are inside site-packages.** Do not write next to them; use explicit
   paths for your own models.
3. **Memory profile differs.** Peak RSS is lower on long files, higher on none.
4. **The `__main__` block in `pyshiro/align.py`** now calls the same `main()` as the
   console script, instead of a separate demo with hard-coded paths.

## Rolling back

The formats are identical, so a model trained with either version works with the other.
To go back:

```bash
pip uninstall pyshiro-fast
pip install git+https://github.com/wavtechyukky/pyshiro
```
