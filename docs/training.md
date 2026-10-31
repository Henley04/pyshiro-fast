# Training

`pyshiro-fast` can train its own `.hsmm` models from a labelled corpus. The algorithm is
the same as upstream: EM over GMM emissions and duration distributions, with a
duration-free bootstrap and optional DAEM annealing.

## Corpus layout

Two directories with matching stems:

```
corpus/
  wav/
    song01.wav      # 16 kHz mono
    song02.wav
  lab/
    song01.lab      # initial boundaries
    song02.lab
```

Labels may be:

- HTK `.lab` in 100 ns units (ENUNU / NNSVS style),
- `.lab` with seconds,
- Praat TextGrid,
- Audacity labels.

The format is detected automatically. Labels only need to be roughly right — training
refines them — but gross errors (missing phonemes, wrong order) will propagate.

## A first model

```bash
pyshiro-train --wav_dir corpus/wav --lab_dir corpus/lab \
    --phonemap my_phonemap.json --out model.hsmm --iters 8
```

`--phonemap` defines the phoneme set, topology and duration indices. Use the bundled
`jp-v2` phonemap as a starting point:

```bash
python -c "import pyshiro; print(pyshiro.bundled_model_path()[1])"
```

## Recommended settings

For a singing-voice corpus of a few hours:

```bash
pyshiro-train --wav_dir corpus/wav --lab_dir corpus/lab \
    --phonemap my_phonemap.json --out model.hsmm \
    --hmm_iters 4 --daem --nmix 4 --iters 16 --cap_relax_iter 6
```

Why each flag:

| Flag | Reason |
|---|---|
| `--hmm_iters 4` | initialise GMMs without duration constraints first; durations estimated from a bad alignment would be wrong |
| `--daem` | anneals the observation likelihood so early iterations do not lock onto a local optimum |
| `--nmix 4` | 1 mixture per state is too coarse for 12-dim MFCC × 3 streams; 4 is a good accuracy/size trade-off |
| `--iters 16` | with `--nmix 4` the model splits twice (at ~⅓ and ~⅔ of the iterations); each split needs iterations to recover |
| `--cap_relax_iter 6` | limits each state to 200 frames early, then relaxes to 1000 for long vowels and pauses |

## Monitoring

Every iteration prints training log-likelihood per frame:

```
=== Iteration 5/16 (hsmm) ===
  train log-likelihood: -71.8342 nats/frame
  test log-likelihood: -74.1092 nats/frame
```

Add `--test_wav_dir` / `--test_lab_dir` to track held-out data. A widening gap between
train and test means overfitting: reduce `--nmix`, reduce `--iters`, or add data.

A per-iteration log is written next to the output model.

## Checkpoints and resuming

`model.iter001.hsmm`, `model.iter002.hsmm`, … are written after each iteration.

```bash
pyshiro-train ... --init_model model.iter006.hsmm --start_iter 6 --iters 16
```

`--start_iter` is 0-based and must match the checkpoint number minus one.

## GMM splitting

With `--nmix 4`, the model starts at 1 mixture and doubles at fixed points, perturbing
the means to break symmetry. Splitting a converged model is much more effective than
starting with 4 mixtures from a flat start.

If you resume from a checkpoint taken after a split, keep `--nmix` at its original value,
or the schedule will split again.

## Cache

Global feature statistics are cached between runs to avoid recomputing them. Delete the
cache file if you change the corpus or the phonemap.

## Aligning with the trained model

```bash
pyshiro-align song.wav song.txt --model model.hsmm --phonemap my_phonemap.json
```

## Triphones

If monophone accuracy is not enough, expand the model to context-dependent triphones
using the contexts observed in the corpus:

```bash
python -m pyshiro.untie \
    --phonemap my_phonemap.json --model model.hsmm --lab_dir corpus/lab \
    --out_phonemap tri_phonemap.json --out_model tri.hsmm
```

Triphone models are several times larger and slower to align with. Measure before
committing to one.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| log-likelihood decreases | label/audio mismatch, or phonemap does not cover the labels |
| all durations collapse to the floor | duration variance estimated from too little data |
| training is very slow | feature extraction dominates; raise `--jobs`, or precompute |
| `br` boundaries drift | duration model for `br` is weak; add `br` to `nodur_phonemes` when aligning |
