# Performance

## Results

All numbers are wall-clock time for the full two-pass alignment (`forced_align_2pass`)
on one x86-64 machine, Python 3.11, numba JIT warm.

| Audio | Frames `T` | States `N` | Upstream | pyshiro-fast | Speedup | Real-time factor | Peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 s | 4 000 | 1 005 | 11.09 s | **0.65 s** | 17× | 31× | 737 / 192 MB |
| 60 s | 12 000 | 3 006 | 251.4 s | **2.50 s** | 101× | 24× | 737 / 324 MB |
| 120 s | 24 000 | 6 006 | 1 986.5 s | **6.79 s** | 293× | 18× | 2 401 / 742 MB |
| 300 s | 60 000 | 15 006 | ≈ 8.7 h † | **91.7 s** | ≈ 341× | 3.3× | ≥ 14 GB ‡ / 255 MB |
| 600 s | 120 000 | 30 006 | ≈ 69 h † | **340.2 s** | ≈ 730× | 1.8× | ≥ 57 GB ‡ / 332 MB |

† Extrapolated from the three measured points with `t ≈ 1.155e-3·s³ + 1.85 s`
(fit error < 0.6 %). Upstream cost is cubic in audio length, so measuring 10 minutes
would take three days.
‡ Computed from the three `(N, T+1)` matrices upstream allocates: `score` float64 +
`back_d` int32 + `back_src` int32 = 16 B per cell.

Peak RSS is written as upstream / pyshiro-fast. It is the maximum resident set of the
whole process, so it includes the interpreter and the feature arrays.

"Real-time factor" is audio duration divided by alignment time: above 1× means faster
than the recording plays.

## Validation

Two kinds of input, for two different questions.

### Exactness and long audio: synthetic features

The scaling table above uses deterministic feature streams sampled from the model's own
GMMs, with known segment boundaries. They serve two purposes: the 10-minute case can be
measured at all (upstream would need about three days), and both implementations can be
fed byte-for-byte identical input, so the comparison is exact rather than statistical.

DP cost depends only on the shape of the lattice, `(T, N)` — the loop structure and the
pruning windows are derived from the state sequence, never from feature values — so the
timing transfers to real audio. Feature extraction is the only extra cost there, and it
is linear.

### Usability: four real GTSinger clips

[GTSinger](https://huggingface.co/datasets/GTSinger/GTSinger) (NeurIPS 2024, CC BY-NC-SA
4.0) Japanese subset, two singers, four techniques. Phone lists come from the corpus
annotation, mapped onto the pyshiro inventory — no lyrics front-end involved. The audio
is **not** redistributed here; `bench/real_audio_check.py` reproduces the numbers against
your own copy. If you cite these figures, cite the corpus as well: Zhang et al.,
*GTSinger: A Global Multi-Technique Singing Corpus*, NeurIPS 2024
([arXiv:2409.13832](https://arxiv.org/abs/2409.13832)).

| Clip | Voice | Audio | Phones | Upstream | pyshiro-fast | Speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Tenor, vibrato | 32.06 s | 69 | 4.81 s | **0.87 s** | 5.5× |
| 2 | Tenor, mixed / falsetto | 27.97 s | 110 | 7.24 s | **0.88 s** | 8.2× |
| 3 | Soprano, vibrato | 21.90 s | 41 | 2.03 s | **0.52 s** | 3.9× |
| 4 | Soprano, breathy | 17.82 s | 41 | 1.64 s | **0.60 s** | 2.7× |

Lattice sizes `(T, N)`: 6 412/207, 5 593/330, 4 380/123, 3 564/123.

All four clips: **every segment and the log-likelihood are bit-identical** between the two
versions (`np.array_equal` on the segment array, `==` on the float64 log-likelihood).

The speedups are smaller than in the table above because these clips are short and few
phonemes: upstream's cost grows with `N·T·D`, and with 41–110 phones the quadratic term
has not taken over yet. At 20 s with 1 005 states upstream already needs 11 s; here 32 s
with 207 states costs 4.8 s.

Two independent sanity checks per clip: obstruent duration, which the acoustics
constrain regardless of any annotation, and the deviation from the corpus' word
boundaries (interior boundaries only).

| Clip | Obstruent median | p90 | max | Word boundary median | Within 50 ms |
|---:|---:|---:|---:|---:|---:|
| 1 | 65 ms | 167 ms | 185 ms | — | — |
| 2 | 60 ms | 162 ms | 175 ms | 67 ms | 45 % |
| 3 | 70 ms | 195 ms | 230 ms | 76 ms | 39 % |
| 4 | 95 ms | 155 ms | 155 ms | 40 ms | 57 % |

Stops, fricatives and affricates stay short on every clip — no consonant swallows a held
note, which is the failure mode you get when an aligner drifts.

Clip 1 has no boundary figure because its own annotation is degenerate: 46 % of its 69
phones sit exactly on the corpus' 30 ms floor and a single phone is 5.17 s, which packs the
first 20 phones into under one second — physically impossible for sung Japanese, and not
something any aligner can match. Its obstruent statistics are still valid, and they are
the reason it is reported rather than dropped. For clips 2–4 the residual is a few tens of
milliseconds, the order of the annotation's own 30 ms quantisation.

### Full CLI on a synthetic waveform

A 40 s WAV generated in NumPy (harmonic stack with vibrato, amplitude modulation and
noise bursts) through the complete CLI — read WAV → MFCC → lyrics → alignment → write
`.lab` — both versions, both including JIT warm-up:

| | Time |
|---|---:|
| upstream | 22.7 s |
| pyshiro-fast | **1.9 s** |

The two `.lab` files agree on every start and end frame (maximum difference: 0), and the
log-likelihood is the same to the last bit. The waveform is synthetic; it exercises the
whole pipeline but says nothing about alignment quality — that is what the GTSinger clips
above are for.

## Why it is faster

The upstream kernel loops over every possible duration `d` for every state and frame,
giving `O(N·T²)`. The first pass — which has no duration prior, so `ceil = T` — is about
96 % of the runtime. Four changes remove that quadratic factor.

### 1. Exact pruning of the duration axis

`d` is only scanned where the predecessor state can actually be:

```text
d ∈ [floor, ceil] ∩ [q − hi(src), q − lo(src)]
```

Cells outside that range hold `-inf`, so this skips work without changing the result.
The second pass benefits most: with a ±60-frame window it becomes effectively
`O(N·window²)`.

### 2. `O(1)` argmax in the duration-free pass

Without a duration prior, the recurrence reduces to a sliding-window maximum:

```text
score[n,t] = cum[t] + max_p( score[src,p] − cum[p] ),   p ∈ [t−ceil, t−floor]
```

The maximum over `p` is maintained incrementally as `t` advances, so each cell costs
`O(1)` instead of `O(t)`: `O(N·T²) → O(N·T)`.

The value is still computed as `score[src,p] + (cum[t] − cum[p])` in the original order,
and `ps`/`cum[p]` are cached when the maximum is updated rather than re-read. That is why
the log-likelihood is bit-identical and not merely close.

### 3. Compact backpointers

Upstream stores three `(N, T+1)` arrays. Here:

- `d` (up to `T`) and the predecessor offset (up to 7 states) are packed into one
  `int32`: `(d << 3) | offset`.
- Scores are kept only for the last `K` rows, where `K` is the longest possible skip
  edge (usually 1–3), in a ring buffer.

16 B per cell becomes 4 B, and the score matrix disappears.

### 4. Hirschberg divide-and-conquer traceback

When the packed backpointer matrix would exceed `backtrace_mem_mb`, the state axis is
split at its midpoint, forward DP runs on the left half, reverse DP on the right half,
and the best meeting frame is chosen. Both halves recurse. Memory becomes `O(K·T)`
instead of `O(N·T)`.

The reverse pass reuses the forward kernel: reverse the state order, reverse the time
axis, flip the adjacency lists, and replace prefix sums with suffix sums. One DP
implementation, not two.

## Cost of the divide-and-conquer path

| 300 s input | Time | Peak RSS |
|---|---:|---:|
| full matrix (`backtrace_mem_mb=4096`) | 45.3 s | ≈ 3.4 GiB (computed) |
| divide-and-conquer (`backtrace_mem_mb=0`) | 91.7 s | 255 MB |

The divide-and-conquer path pays `log N` passes, so it is 2–2.5× slower. It is chosen
automatically only when the matrix would not fit in the budget. If you have RAM and want
speed, raise the budget:

```python
segments, loglik = pyshiro.forced_align_2pass(
    model, streams, state_seq, backtrace_mem_mb=4096)
```

Both paths return identical segments; this is checked by the test suite.

## Tuning

| Goal | Setting |
|---|---|
| fastest on a machine with RAM | `backtrace_mem_mb=4096` (or higher) |
| lowest memory, speed irrelevant | `backtrace_mem_mb=0` |
| default | `1024` — full matrix up to ~250 M cells, then divide-and-conquer |
| long files on a small container | `backtrace_mem_mb=256` |

Rough memory for the full-matrix path: `4 · N · (T+1)` bytes — about 2.3 GiB per
10 000 states at 300 s. Add ~40 MB for features and the DP context.

## Reproducing these numbers

The benchmark script is in the repository:

```bash
# scaling sweep with deterministic synthetic features
python bench/bench_align_speed.py --sweep 10,20,60,120,300 \
    --model pyshiro/data/models/pyshiro-jp-v2.hsmm \
    --phonemap pyshiro/data/models/pyshiro-jp-v2_phonemap.json

# real audio through the CLI path
python bench/bench_align_speed.py --wav song.wav --lyric song.txt \
    --model pyshiro/data/models/pyshiro-jp-v2.hsmm \
    --phonemap pyshiro/data/models/pyshiro-jp-v2_phonemap.json

# one recording, with sanity statistics (and an optional reference phone timing)
python bench/real_audio_check.py --wav clip.wav \
    --phones "pau,a,k,a,i,g,u,ts,u,pau" \
    --ref-durs "0.30,0.05,0.12,0.40,0.08,0.15,0.07,0.20,0.35,0.25"
```

`real_audio_check.py` is what produced the GTSinger tables above; it takes a plain phone
list, so it works with any annotated corpus.

The sweep uses features sampled deterministically from the model's own GMMs, which is what
makes the exact comparison in the first section possible. Real audio adds feature
extraction — linear, and well under a second for a few minutes.

## What is not optimised yet

- The kernel is memory-bound at roughly 90–130 M cells/s. A tiled sweep (time outer,
  states inner, blocking both) would improve cache locality and could give another
  3–5×, at the cost of rewriting the inner loop.
- The duration pass still scans a window of durations. A monotone-queue formulation
  exists for the Gaussian duration prior but was not implemented, because the second
  pass is already a small fraction of the total.
- Feature extraction is not parallelised. For batch corpora, `--jobs` parallelises
  training only.

Contributions here are welcome as long as `tests/test_viterbi_exact.py` stays green.
