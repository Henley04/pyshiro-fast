# pyshiro-fast

Fast HSMM forced alignment for Japanese singing voice. A drop-in fork of
[pyshiro](https://github.com/wavtechyukky/pyshiro) that aligns long recordings
100–700× faster, without changing the result.

```bash
pip install pyshiro-fast
```

```python
import pyshiro

model, phonemap = pyshiro.load_bundled_model()
streams  = pyshiro.extract_mfcc_from_file("song.wav")
phonemes = pyshiro.convert_lyric_file("song.txt")

T = streams[0].shape[0]
state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
segments, loglik = pyshiro.forced_align_2pass(
    model, streams, state_seq, nodur_phonemes={"pau", "br"})

intervals = pyshiro.segments_to_phoneme_intervals(phonemes, segments)
pyshiro.write_lab(intervals, "song.lab")
```

## What it does

Given a 16 kHz mono WAV and a lyrics file, `pyshiro-fast` returns the start and end
frame of every phoneme. It uses the same two-pass scheme as the original SHIRO
toolkit: a duration-free HMM pass for a rough path, then an HSMM pass with explicit
duration distributions for the final boundaries. Typical uses are annotation for
singing-voice databases, label conversion, and preparing training data for
NNSVS / ENUNU-style singing synthesizers.

## Performance

| Audio length | Upstream | pyshiro-fast | Speedup | Peak RSS |
|---:|---:|---:|---:|---:|
| 20 s | 11.09 s | 0.65 s | 17× | 737 / 192 MB |
| 60 s | 251.4 s | 2.50 s | 101× | 737 / 324 MB |
| 120 s | 1 986.5 s | 6.79 s | 293× | 2 401 / 742 MB |
| 300 s | ≈ 8.7 h † | 91.7 s | ≈ 341× | ≥ 14 GB / 255 MB |
| 600 s | ≈ 69 h † | 340.2 s | ≈ 730× | ≥ 57 GB / 332 MB |

† extrapolated from three measured points; upstream is cubic in audio length.
Peak RSS is shown as upstream / pyshiro-fast. Details and measurement method:
[Performance](performance.md).

The speedup comes from four changes to the decoder, all in `pyshiro/align.py`:

- exact pruning of the duration axis against the predecessor's reachable window
- `O(1)` sliding-window argmax in the duration-free pass (`O(N·T²) → O(N·T)`)
- backpointers packed into one `int32` per cell, scores kept in a `K`-row ring buffer
- Hirschberg divide-and-conquer traceback above a configurable memory threshold

None of them changes arithmetic order, so segments and log-likelihood are identical to
upstream — verified on 13 configurations against the original kernel, which is kept in
the source as `_viterbi_dp`.

## Start here

| If you want to… | Read |
|---|---|
| install the package | [Installation](installation.md) |
| align a song in five minutes | [Quick start](quickstart.md) |
| use the command line | [CLI reference](cli.md) |
| call the library from Python | [Python API](api.md) |
| understand the speedup or tune memory | [Performance](performance.md) |
| replace an existing pyshiro install | [Migrating from pyshiro](migration.md) |
| train your own model | [Training](training.md) |
| fix a broken alignment | [FAQ](faq.md) |

## Bundled models

Two Japanese models ship inside the wheel, so a fresh install can align immediately.

| Name | Notes |
|---|---|
| `jp-v2` (default) | 4-mixture GMMs; mean boundary error 2.05 frames on held-out singers |
| `jp-v1` | 19.4 h multi-database model; more stable on `br` (breath) |

```python
model, phonemap = pyshiro.load_bundled_model("jp-v1")
```

Credits and training data: [Models](models.md).

## Requirements

Python 3.10–3.13, `numpy`, `scipy`, `soundfile`, `numba`, `msgpack`. No GPU.
Linux, macOS and Windows are supported wherever numba runs.

## License

GPL-3.0-only, inherited from upstream pyshiro and SHIRO. See `LICENSE` and `NOTICE`
in the repository.
