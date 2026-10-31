# Python API

Everything below is available from the top-level package:

```python
import pyshiro
```

Frame rate is 200 fps (5 ms per frame); `pyshiro.align.FPS == 200`.

---

## Models

### `load_hsmm(path) -> HSMMModel`

Reads a MessagePack `.hsmm` file.

```python
model = pyshiro.load_hsmm("my_model.hsmm")
model.nstate      # number of GMMs
model.nstream     # 3 (MFCC, Δ, ΔΔ)
model.ndim        # 12 coefficients per stream
model.nduration   # number of duration distributions
model.streams     # list[Stream], each with .gmms
model.durations   # list[Duration], each with .mean, .var, .floor, .ceil
```

### `save_hsmm(model, path)`

Writes a model in the same format; round-trips are lossless.

### `load_bundled_model(name="jp-v2") -> (HSMMModel, dict)`

Loads a model that ships with the package and its phonemap. `name` is `"jp-v2"`
(default) or `"jp-v1"`.

```python
model, phonemap = pyshiro.load_bundled_model("jp-v1")
```

### `bundled_model_path(name="jp-v2") -> (Path, Path)`

Returns `(hsmm_path, phonemap_path)` without loading. Useful for passing paths to the
CLI or to `pyshiro-train`.

```python
hsmm, pm = pyshiro.bundled_model_path()
```

### `pyshiro.BUNDLED_MODELS`

Tuple of valid names: `("jp-v2", "jp-v1")`.

---

## Features

### `extract_mfcc(audio, sr=16000) -> list[np.ndarray]`

Takes a mono `float` array in `[-1, 1]`, returns three `(T, 12)` arrays: MFCC, Δ, ΔΔ.
The front end matches SHIRO/ciglet, so models trained with the original toolkit work
unchanged.

```python
import soundfile as sf
audio, sr = sf.read("song.wav")          # must be 16 kHz mono
streams = pyshiro.extract_mfcc(audio, sr)
```

### `extract_mfcc_from_file(wav_path) -> list[np.ndarray]`

Reads the WAV with `soundfile` and calls `extract_mfcc`. No resampling: a 44.1 kHz file
produces features at the wrong rate and meaningless boundaries.

---

## Phonemes

### `load_table(table_path=None) -> dict`

Loads the kana-to-phoneme table. With no argument, uses the table bundled with the
package.

### `text_to_phonemes(text, table) -> list[str]`

Converts one string of hiragana (with optional inline tokens such as `br`) to a phoneme
list. Unknown characters are skipped with a warning.

### `convert_lyric_file(txt_path, table=None) -> list[str]`

Reads a lyrics file and returns the full phoneme sequence, inserting `pau` at every line
break. If the file already contains ASCII phonemes, it is parsed as a phoneme sequence
instead of being converted.

```python
phonemes = pyshiro.convert_lyric_file("song.txt")
```

---

## Alignment

### `build_state_sequence(phonemes, phonemap, T) -> list[StateInfo]`

Expands the phoneme list into three states per phoneme, using the phonemap for topology
and duration indices. `T` is the number of frames; it is used to validate that the
sequence can fit the audio.

```python
T = streams[0].shape[0]
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
```

Each `StateInfo` has `phoneme`, `state_idx` (0/1/2), `dur_idx`, `out_idx`.

### `forced_align(...) -> (segments, loglik)`

Single Viterbi pass.

```python
segments, loglik = pyshiro.forced_align(
    model, streams, state_seq,
    use_duration=True,          # False = HMM mode, no duration prior
    initial_segments=None,      # optional [(start, end)] per state, from a first pass
    window=60,                  # search width around initial_segments, in frames
    daem_temp=1.0,              # scale observation log-likelihood (training only)
    nodur_phonemes=None,        # e.g. {"pau", "br"} to ignore their duration models
    hmm_cap=None,               # max frames per state in HMM mode
    backtrace_mem_mb=1024,      # switch to divide-and-conquer above this size
)
```

Returns `segments` — one `(start, end)` per **state**, `end` exclusive — and `loglik`,
the total log-likelihood in nats. Divide by `T` to compare across files.

`backtrace_mem_mb=0` forces the divide-and-conquer traceback (slower, minimal memory);
a large value forces the full matrix (faster, `O(N·T)` memory). Both produce the same
output.

### `forced_align_2pass(...) -> (segments, loglik)`

The pipeline used by the CLI: HMM pass to get a rough path, HSMM pass restricted to
`±window` frames around it. Same keyword arguments, plus `window`.

```python
segments, loglik = pyshiro.forced_align_2pass(
    model, streams, state_seq, window=60, nodur_phonemes={"pau", "br"})
```

Prefer this over a single HSMM pass: it is both faster and more robust.

### `refine_pau_boundaries(intervals, audio, sr, ...) -> intervals`

Post-processes `pau` → soft-onset consonant boundaries (nasals, approximants,
fricatives) with short-time RMS onset detection. HSMM alignment tends to place those
boundaries slightly inside the silence; this moves them to the energy rise.
Input and output are in **frames** (`segments_to_phoneme_intervals` output).

```python
intervals = pyshiro.segments_to_phoneme_intervals(phonemes, segments)
intervals = pyshiro.refine_pau_boundaries(intervals, audio, sr)
```

| Argument | Default | Meaning |
|---|---|---|
| `soft_onset_phones` | `SOFT_ONSET_PHONES` | phonemes to adjust |
| `search_sec` | `0.05` | search radius around the boundary |
| `rms_hop_sec` | `0.0025` | RMS frame hop |
| `threshold_ratio` | `0.15` | fraction of consonant peak RMS |
| `min_pau_frames` | `2` | skip shorter pauses |

### `realign_external_labels(...) -> intervals`

Converts labels produced by another tool into this model's conventions. Long vowels and
pauses are used as anchors; each segment between anchors is re-aligned with the HSMM.

```python
intervals = pyshiro.realign_external_labels(
    model, streams, external_intervals, phonemap,
    anchor_vowel_min_sec=0.5,
    anchor_pau_min_sec=0.5,
    max_seg_sec=30.0,
    ignore_transitions=None,      # set of (left, right) phoneme pairs to leave alone
    phoneme_map=None,             # rename symbols before aligning
    unknown_phoneme_map=None,     # fallback for symbols absent from the phonemap
)
```

`intervals` must be in **seconds** (as returned by `read_lab`, `read_textgrid` or
`read_audacity`). The return value is in **frames**, ready for `write_lab` and the other
writers.

### `pyshiro.SILENCE_PHONES`

`frozenset({"pau", "sil", "cl", "br"})` — phonemes treated as silence by the alignment
helpers.

---

## Labels

All writers accept frame indices (int) or seconds (float) and detect which they got.

### `segments_to_phoneme_intervals(phonemes, segments) -> list[(start, end, phoneme)]`

Collapses the per-state segments (3 per phoneme) into per-phoneme intervals, in frames.

```python
intervals = pyshiro.segments_to_phoneme_intervals(phonemes, segments)
```

### Writers

| Function | Output |
|---|---|
| `write_lab(intervals, path)` | HTK `.lab`, 100 ns integers |
| `write_lab_sec(intervals, path)` | `.lab`, space-separated seconds |
| `write_textgrid(intervals, path, tier_name="phoneme")` | Praat TextGrid |
| `write_audacity(intervals, path)` | tab-separated seconds |

### Readers

| Function | Returns |
|---|---|
| `read_lab(path)` | `[(start_sec, end_sec, label)]`, auto-detects HTK vs seconds |
| `read_textgrid(path, tier_name=None)` | `[(start_sec, end_sec, label)]` |
| `read_audacity(path)` | `[(start_sec, end_sec, label)]` |

```python
intervals = pyshiro.read_lab("song.lab")        # seconds
```

---

## End-to-end example with a post-processed boundary

```python
import soundfile as sf
import pyshiro

model, phonemap = pyshiro.load_bundled_model()
audio, sr = sf.read("song.wav")
streams = pyshiro.extract_mfcc(audio, sr)

phonemes = pyshiro.convert_lyric_file("song.txt")
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, streams[0].shape[0])

segments, loglik = pyshiro.forced_align_2pass(
    model, streams, state_seq, nodur_phonemes={"pau", "br"})

intervals = pyshiro.segments_to_phoneme_intervals(phonemes, segments)
intervals = pyshiro.refine_pau_boundaries(intervals, audio, sr)
pyshiro.write_lab(intervals, "song.lab")
print(f"{loglik / streams[0].shape[0]:.3f} nats/frame")
```

## Errors

| Exception | When |
|---|---|
| `RuntimeError("alignment failed: no valid path found…")` | no state path fits the audio; lyrics too long or phonemap mismatch |
| `ValueError("skip edge spans …")` | a `pskip` edge longer than 7 states; unsupported by the packed backpointer |
| `FileNotFoundError` from `bundled_model_path` | package data missing (source checkout without models) |
