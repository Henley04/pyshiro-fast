# FAQ

## Alignment

### `RuntimeError: alignment failed: no valid path found`

No state sequence covers the audio. Causes, in order of likelihood:

1. **Lyrics longer than the audio.** Every phoneme needs at least one frame and pauses
   have floors too. Check `T` (frames) against the phoneme count: `pyshiro-align` prints
   both. Rough guide: one mora ≈ 150–250 ms, so 200 phonemes need at least ~60 s.
2. **Wrong phonemap.** A phonemap that lacks one of your phonemes makes it impossible to
   place.
3. **Truncated or silent audio.** A zero-length WAV, or one shorter than a few frames.

### Boundaries are shifted by a constant offset

The audio is not 16 kHz, or not mono. `extract_mfcc` does not resample.

```bash
ffmpeg -i in.wav -ac 1 -ar 16000 out.wav
```

### One phoneme spans several seconds

Usually a line break is missing in the lyrics: each line becomes one `pau`, and long
phrases let the model stretch a vowel. Split the phrase with line breaks.

### `pau` boundaries sit inside the consonant

Known HSMM behaviour for soft onsets (nasals, approximants, fricatives). Apply the
onset-detection post-process:

```python
intervals = pyshiro.refine_pau_boundaries(intervals, audio, sr)
```

### Breath (`br`) regions are too short

The `jp-v2` model under-estimates `br`. Use `jp-v1` for material with many breaths:

```python
model, phonemap = pyshiro.load_bundled_model("jp-v1")
```

Or keep `br` out of the duration model when aligning: `nodur_phonemes={"pau","br"}`.

### Results differ slightly from upstream

They should not — differences indicate a bug. Please open an issue with the input that
reproduces it. On very large inputs, make sure both runs used the same
`backtrace_mem_mb` budget only for timing; the output must not depend on it.

## Performance

### The first run is slow

numba compiles the kernel on first use (10–20 s). Later runs reuse the cache in
`__pycache__`. In containers with a read-only package directory, set
`NUMBA_CACHE_DIR=/tmp/numba`.

### Memory is higher than the table says

Peak RSS includes Python, numpy, the features (three `(T, 12)` arrays), and the
backpointer matrix. Lower `backtrace_mem_mb` to force the divide-and-conquer path:

```python
pyshiro.forced_align_2pass(..., backtrace_mem_mb=256)
```

### Is it faster to use one pass?

`forced_align(..., use_duration=True)` alone is not recommended: without a first-pass
window, the duration scan covers the whole timeline and is much slower than the two-pass
pipeline for the same accuracy.

## Installation

### `ModuleNotFoundError: pyshiro`

You are running a different interpreter than the one you installed into:

```bash
python -m pip install pyshiro-fast
python -m pyshiro.align --help
```

### Both `pyshiro` and `pyshiro-fast` are installed

Both provide the `pyshiro` module. Remove the upstream one:

```bash
pip uninstall pyshiro
pip install --force-reinstall pyshiro-fast
```

### `ImportError: cannot import name 'types' from 'llvmlite'`

`numba` and `llvmlite` versions do not match. Reinstall them together:

```bash
pip install --force-reinstall "numba>=0.60"
```

### `ValueError: skip edge spans N states (limit 7)`

Your phonemap has a `pskip` edge spanning more than 7 states, which the packed
backpointer cannot encode. Either shorten the skip in the phonemap or use upstream's
kernel for that model.

## Data and output

### What lyric format is accepted?

Hiragana, UTF-8, one phrase per line. Line breaks become `pau`. Inline `br` is allowed.
Katakana is converted through the bundled table. ASCII phoneme sequences are passed
through unchanged.

### Why is my `.lab` in huge integers?

HTK format: 100 ns units. Frame `n` is `n × 5 000 000`. Use `write_lab_sec` for seconds.

### Can I use a TextGrid made in Praat?

Yes — `pyshiro.read_textgrid`, then `realign_external_labels` to convert its conventions
to this model's. See [API](api.md#realign_external_labels).

### Does it work for speech, not singing?

The bundled models are trained on singing voice. The algorithm is generic, but for speech
you need a model trained on speech with a matching phonemap.

## Licensing

### Can I use this commercially?

The code is GPL-3.0. The bundled models are redistributed from upstream under the same
license, but they were trained on third-party singing databases — check those terms
before commercial use. See [Models](models.md).
