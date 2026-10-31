# Quick start

This page takes a WAV file and a lyrics file to a `.lab` file, then shows how to check
the result. It assumes `pip install pyshiro-fast` is done.

## 1. Prepare the input

**Audio.** 16 kHz mono WAV. Anything else must be converted first:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 song.wav
```

**Lyrics.** A UTF-8 text file in hiragana, one phrase per line. Each line break becomes
a `pau` (pause). Inline tokens are allowed: `br` for a breath, katakana is converted
through the bundled table.

```
あかいぐつ
あるいてた
ゆきのあさ
```

Phoneme sequences written in ASCII (for example `a k a i g u t s u`) are also accepted
and used as-is.

## 2. Align

### Command line

```bash
pyshiro-align song.wav song.txt
```

With no `--model`, the bundled `jp-v2` model is used. The output is written next to the
WAV as `song.lab`. Progress and the average log-likelihood are printed:

```
Loading model: .../pyshiro/data/models/pyshiro-jp-v2.hsmm
Extracting features: song.wav
Converting lyrics: song.txt
  phonemes (175): pau a k a i ...
Aligning... (T=8000 frames, 175 phonemes)
  log-likelihood: -75.0469 nats/frame
Wrote: song.lab
```

### Python

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
print(f"{loglik / T:.4f} nats/frame")
```

## 3. Read the result

`.lab` uses the HTK convention: integer time units of 100 ns, three columns.

```
0 12500000 pau
12500000 15250000 a
15250000 16500000 k
...
```

Frame indices are converted at 200 fps (5 ms per frame), so frame `n` is
`n × 5 000 000` in 100 ns units.

To work in seconds instead:

```python
from pyshiro.labels import segments_to_phoneme_intervals, write_lab_sec

intervals = segments_to_phoneme_intervals(phonemes, segments)   # frames
write_lab_sec(intervals, "song_seconds.lab")
```

`write_textgrid(intervals, "song.TextGrid")` and `write_audacity(intervals, "song.txt")`
produce the other two formats. All writers accept frame indices (int) or seconds
(float) and detect which one they were given.

## 4. Check the result

Two numbers tell you whether the alignment is plausible.

**Log-likelihood per frame.** Compare against a known-good file aligned with the same
model. A value far below it (more negative) usually means the lyrics do not match the
audio, or the audio is not 16 kHz.

**Phoneme durations.** Japanese phonemes are short — vowels typically 80–400 ms,
consonants 30–150 ms. A consonant longer than a second, or a vowel under 20 ms, points
at a misalignment; look at that region in a waveform editor.

```python
FPS = 200
for start, end, ph in intervals:
    dur_ms = (end - start) / FPS * 1000
    if ph not in ("pau", "br") and (dur_ms < 20 or dur_ms > 800):
        print(f"{ph}: {dur_ms:.0f} ms at {start / FPS:.2f}s")
```

## 5. Where to go next

- Adjusting `pau`/consonant boundaries: `pyshiro.refine_pau_boundaries` — see
  [Python API](api.md).
- Converting labels made by another tool: `pyshiro.realign_external_labels` — see
  [Migrating from pyshiro](migration.md).
- Long files: nothing to do. Memory is capped by `backtrace_mem_mb` and the traceback
  switches to divide-and-conquer automatically. See [Performance](performance.md).
- Training a model on your own corpus: [Training](training.md).

## Common mistakes

| Mistake | Effect |
|---|---|
| stereo or 44.1 kHz input | feature mismatch, meaningless boundaries |
| romaji lyrics | not converted; use hiragana or ASCII phonemes |
| lyrics longer than the audio | `alignment failed: no valid path found` |
| one very long line | one long `pau` guess; split phrases with line breaks |
| text file with BOM | first phoneme may be wrong; save as UTF-8 without BOM |
