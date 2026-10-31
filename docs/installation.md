# Installation

## Requirements

| | |
|---|---|
| Python | 3.10, 3.11, 3.12, 3.13 |
| OS | Linux, macOS, Windows (any platform numba supports) |
| Hardware | CPU only; no GPU required |
| Audio | 16 kHz mono WAV (other sample rates are read by `soundfile` and must be resampled beforehand) |

Runtime dependencies are installed automatically: `numpy`, `scipy`, `soundfile`,
`numba`, `msgpack`.

## From PyPI

```bash
pip install pyshiro-fast
```

Check that it imports:

```bash
python -c "import pyshiro; print(pyshiro.__version__)"
```

The wheel includes both pre-trained models (~650 KB) and the kana-to-phoneme table,
so no extra download is needed.

## From source

```bash
git clone https://github.com/henley04/pyshiro-fast
cd pyshiro-fast
pip install -e .
```

For development (tests, build tooling):

```bash
pip install -e ".[dev]"
pytest
```

## Verifying the install

```python
import pyshiro

model, phonemap = pyshiro.load_bundled_model()
print(model.nstate, len(phonemap))     # 123 49
print(pyshiro.bundled_model_path())    # absolute paths inside site-packages
```

Running an alignment for the first time triggers numba JIT compilation and takes
10–20 s longer than subsequent runs. The compiled kernels are cached in
`__pycache__` next to the source, so the cost is paid once per environment.

## Import name

The distribution is `pyshiro-fast`; the import name is `pyshiro`:

```python
import pyshiro          # recommended
import pyshiro_fast     # alias, same objects
```

If you have the upstream `pyshiro` installed from source in the same environment,
uninstall it first — both provide the `pyshiro` module.

```bash
pip uninstall pyshiro
pip install pyshiro-fast
```

## Platform notes

**Apple Silicon / arm64.** numba and `llvmlite` need a recent version; `pip install -U
numba` resolves the correct wheel. If you see `ImportError: cannot import name
'types' from 'llvmlite'`, your `llvmlite` and `numba` versions are mismatched —
reinstall both together.

**Windows.** `soundfile` bundles libsndfile in its wheels; no separate install is
needed. Long paths can break numba's cache; set `NUMBA_CACHE_DIR` to a short path if
compilation fails with a path-length error.

**Containers.** The JIT cache is written next to the package. On read-only images, set
`NUMBA_CACHE_DIR=/tmp/numba` to avoid a permission error on first run.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: pyshiro` | installed under a different interpreter | use the same `python -m pip` you run your script with |
| `alignment failed: no valid path found` | lyrics longer than the audio, or wrong phonemap | check the phoneme count against `T`; see [FAQ](faq.md) |
| very slow first call | numba JIT compilation | expected once per environment |
| `ValueError: skip edge spans … states` | phonemap has a `pskip` edge longer than 7 states | not supported by the packed backpointer; use the upstream kernel for that model |
| garbage boundaries | audio is not 16 kHz or not mono | resample to 16 kHz mono first |
